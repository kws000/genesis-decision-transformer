"""
figure_eight_follow.py
----------------------
Simple car -> Figure-Eight path following demo (Genesis 0.3.0.dev0)
"""

import math
from math import radians
import threading
import time
from pathlib import Path
import numpy as np
import pkgutil, inspect
from genesis.utils.geom  import euler_to_quat
import genesis as gs
import pandas as pd
import random
import torch
from utils.trajectory_utils import yaw_to_sin_cos
from utils.trajectory_utils import sin_cos_to_yaw

# === 行動クローンモデルの読み込み ===


class ControlMLP(torch.nn.Module):

    # ９次元に拡張し順序を合わせる
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=2):
#    def __init__(self, input_dim=6, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 1) 生成する 8 の字 Waypoints
# ---------------------------------------------------------------------------

def generate_bernoulli_waypoints(
        num_points: int = 600,      # resolution of the full ∞
        a: float = 0.5,             # the “a” from (x²+y²)² = a²(x²−y²)
        center=np.zeros(3)          # optional offset of the whole curve
) -> np.ndarray:
    """
    Return (N, 3) way-points for a Bernoulli figure-eight centered at `center`.
    The curve lies in the XY-plane (Z = 0).

    Parametric form  (t ∈ (−π/2,  π/2)  for the right lobe):
        x =  a·√2·cos t / (1 + sin² t)
        y =  a·√2·cos t·sin t / (1 + sin² t)

    We sample that half-lobe, then mirror it to obtain the left half,
    giving a closed, symmetric ∞.
    """
    # sample one half-lobe (avoid end-point singularities)
    t = np.linspace(-math.pi/2 + 1e-3,
                     math.pi/2 - 1e-3,
                     num_points // 2,
                     endpoint=False)

    x_half =  a * math.sqrt(2) * np.cos(t) / (1 + np.sin(t)**2)
    y_half =  a * math.sqrt(2) * np.cos(t) * np.sin(t) / (1 + np.sin(t)**2)

    # mirror to make the other lobe
    x_full = np.concatenate([ x_half, -x_half ])
    y_full = np.concatenate([ y_half, -y_half ])

    pts = np.stack([x_full, y_full, np.zeros_like(x_full)], axis=1)
    return pts + center

# False:PurePursaitによる運転と教師CSVの収集、True:bc_modelによる推論運転
is_mode_ai = False#True
# ビュアーやSleepをスキップする高速モード
is_mode_fast = True#False

# 経路情報
WAYPOINTS = generate_bernoulli_waypoints(a=2.0) # 0.5 m matches the OBJ

# アクセル制御パラメータ
TARGET_SPEED   = 8.0#5.0#3.0#15.0#1.5      # m/s: 巡航目標
KP_SPEED       = 0.1#0.006#1.0         #    : 車速 P 利得
KI_SPEED       = 0.006#0.001#0.0         #    : （必要なら）積分利得  ※これがないとカーブで推進力が足りなくなる
FORCE_CLIP     = 1.5#0.5

# Pure-Pursuit + フィルタ用パラメータ
K_LOOK = 1.0#2.0#1.0#1.5#0.6#1.2           # ルックアヘッド・タイムスケール [s]
V_EPS = 0.5#0.1            # 最低速度下限 [m/s]
MAX_STEER_RAD = 3.1415926535 * 80.0 / 180    # ステア最大角度

# LatencyFilter 用パラメーター
FILTER_TAU = 0.15      # フィルタ時定数 [s]
CONTROL_DT = 0.02      # 制御ループ周期 [s]。1/50Hz くらい

# vは正規化不要とのこと
#roll pitch yaw の順は ROS では標準
def vector_to_euler(v):
    x, y, z = v
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(-z, np.sqrt(x**2 + y**2))
    roll = 0  # ロールは定義できない（方向ベクトルだけでは不定）
    return np.degrees([roll, pitch, yaw])

def set_car_start_pos(car,waypoint_idx: int,waypoint_direc: int):

    # 車の初期位置
    target_wp = get_wp_position(waypoint_idx,WAYPOINTS)
    target_next_wp = get_wp_position(waypoint_idx+waypoint_direc,WAYPOINTS)

    segment_vec = target_next_wp - target_wp
    segment_norm = np.linalg.norm(segment_vec)

    CAR_START_POS_LENGTH = 2.5

    if segment_norm > 0.0:

        # セグメント方向を延長
        segment_vec = CAR_START_POS_LENGTH * segment_vec / segment_norm

        # 車位置
        start_pos = target_wp - segment_vec
        car.set_pos(( start_pos[0], start_pos[1], 0.30 ))   # 8 の字左端スタート

        # 車方向
        angle = vector_to_euler((segment_vec[0],segment_vec[1],0.0))
        quat = euler_to_quat(angle)
        car.set_quat(quat)
    else:
        car.set_pos(( -3.8, 0.0, 0.30 ))   # 8 の字左端スタート
        quat = euler_to_quat((0.0, 0.0, 0.0))
        car.set_quat(quat)



def check_waypoint_passed(pos: np.ndarray, waypoints: np.ndarray, current_wp_idx: int,waypoint_direc: int, threshold: float = 1.0):
    """
    車の現在位置が現在のウェイポイントを通過したかを判定

    Parameters:
        pos: 現在の車の位置 (x, y)
        waypoints: ウェイポイントのリスト（各要素はnp.array([x, y])）
        current_wp_idx: 現在ターゲットにしているウェイポイントのインデックス
        threshold: 通過とみなす距離の閾値 [m]

    Returns:
        passed (bool), new_wp_idx (int)
    """

    wp = waypoints[current_wp_idx]
    distance = np.linalg.norm(pos - wp[:2])

    if distance < threshold:
        # ラップ    
        next_wp_idex = (current_wp_idx + waypoint_direc) % len(waypoints)
        return True, next_wp_idex  # 通過とみなして次へ
    else:
        return False, current_wp_idx

def get_wp_position(wp_idx: int,waypoints: np.ndarray):

    wp_idx = wp_idx % len(waypoints)

    return waypoints[wp_idx][:2]

def find_target_wp_ordered(pos_xy: np.ndarray, waypoints: np.ndarray, lookahead: float, current_idx: int):
    """
    Pure Pursuit に適した、waypoint を順に消化する安定版。

    Returns:
        (new_idx, target_xy)
    """

    #一周したので戻す
    if current_idx >= len(waypoints) - 1:
        current_idx = 0

    acc_dist = 0.0
    next_idx = current_idx
    last_point = pos_xy


    # lookaheadを消化しきるまで回す
    for i in range(current_idx, len(waypoints)):

        # 課題
        # lookaheadが大きいと、インデックスがどんどん消化されてしまう
        # かといって進めないと通過済のインデックスに引っ張られる
        next_idx = i

        p1 = last_point
        p2 = waypoints[i][:2]
        v_p1_to_p2 = p2 - p1
        v_p1_to_p2_size = np.linalg.norm(v_p1_to_p2)

        acc_dist += v_p1_to_p2_size
        if acc_dist >= lookahead:
            # はみ出し分
            over_dist = acc_dist - lookahead

            # 区間の現在割合
            ratio = (v_p1_to_p2_size - over_dist) / v_p1_to_p2_size

            # 消化しきったのではみ出し分をカット
            target = p1 + v_p1_to_p2 * ratio

#　何故か p2 を追いかけるシンプル動作の方が綺麗に旋回するので困った            
#            target = p2

            return next_idx , target

        # p2でも消化できなかったので、インデックス更新
        last_point = p2

    # 消化できなかったので最初から
    next_idx = 0
    return next_idx , waypoints[0][:2]



# ルックアヘッド距離の算出
def compute_lookahead(v: float,
                      k_la: float = 1.0,
                      v_eps: float = 0.1) -> float:
    """
    速度依存ルックアヘッド距離 L を返す
    - v     : 現在速度 [m/s]
    - k_la  : ルックアヘッド・タイムスケール [s]
    - v_eps : 最低速度下限 [m/s]
    """
    return k_la * max(v, v_eps)

# ステア角への一階遅れフィルタ
class LatencyFilter:
    def __init__(self, tau: float, dt: float):
        """
        一階遅れフィルタを表すクラス
        - tau : 時定数 [s]
        - dt  : 制御周期   [s]
        """
        self.tau = tau
        self.dt = dt
        self.alpha = self.dt / (self.tau + self.dt)
        self.prev = 0.0   # 前回出力 δ_out(t−dt)

    def __call__(self, delta_cmd: float) -> float:
        """
        δ_cmd: 生のステア角 [rad]
        戻り値: 平滑化後のステア角 δ_out [rad]
        """
        self.prev = self.prev + self.alpha * (delta_cmd - self.prev)
        return self.prev


# ── Pure-Pursuit 用関数 ──
def pure_pursuit_steer(target_wp: np.ndarray,
                       pos_xy: np.ndarray,
                       yaw: float,
                       lookahead: float,
                       wheelbase: float = 0.3
                       ) -> float:
    """
    Pure-Pursuit 制御でステア角を返す。
    - waypoints : (N,2) の numpy 配列（Z 列が不要なので XY だけでも可）
    - pos_xy    : (2,) の numpy 配列。現在の車両 XY 座標
    - yaw       : 車両の向き（ヨー角）[rad]
    - lookahead : ルックアヘッド距離 L [m]
    - wheelbase : 車両ホイールベース長 [m]
    """

    # 2) 車両座標系→ローカル XY に変換
    dx = target_wp[0] - pos_xy[0]
    dy = target_wp[1] - pos_xy[1]
    # ワールド→車両座標回転：（cosθ sinθ; -sinθ cosθ）
    local_x =  math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

    # 3) Pure-Pursuit 式: δ = atan2(2·wheelbase·sinα, L)
    #    α = atan2(y_local, x_local)
    if local_x == 0 and local_y == 0:
        return 0.0
    alpha = math.atan2(local_y, local_x)
    delta_pp = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)

    # 物理的に許容できる角度に制限
    delta_pp = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta_pp))

    return delta_pp



def get_obs(car_pos,car_vel,car_yaw,target_wp,target_next_wp,passed,is_first_check_point):

    # ターゲット方向
    dx, dy = target_wp - car_pos
    target_yaw = math.atan2(dy, dx)

    # セグメント方向
    segment = target_next_wp - target_wp

    # ヘディング誤差（-pi ～ +pi に wrap）
    # ※車体とターゲット方向の角度差
    heading_error = target_yaw - car_yaw
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    # CTE（ターゲット方向に直交する方向への距離）
    target_dir = target_wp - car_pos
    target_dir = target_dir / np.linalg.norm(target_dir)
    segment_dir = segment / np.linalg.norm(segment)
    inner_angle = np.dot(target_dir, segment_dir)
    # 1.0 0.0 -1.0
    # ↓ x 1.0
    # -1.0 0.0 1.0
    # ↓ + 1.0
    # 0.0 1.0 2.0   ※真正面で0.0　真横で1.0 真後ろで2.0
    perp_error = 1.0 - inner_angle

    #まだチェックポイント上に乗っていないのでコースアウトは無視する
    if is_first_check_point:
        perp_error = 0.0

    
# 断続しないヨー角 １０次元に
    car_yaw_sin,car_yaw_cos = yaw_to_sin_cos(car_yaw)
    return np.array([target_wp[0], target_wp[1], car_pos[0], car_pos[1], car_yaw_sin,car_yaw_cos, car_vel, perp_error, heading_error,passed], dtype=np.float32)
#    # ９次元に拡張し順序を合わせる　要素が間違っている
#    return np.array([target_wp[0], target_wp[1], car_pos[0], car_pos[1], car_yaw, car_vel, perp_error, heading_error,passed], dtype=np.float32)

def compute_reward(obs,t):
    # obs = [x, y, yaw, speed, cross_track_err, heading_err]
    speed = obs[5+1]
    cte   = obs[6+1]  # Cross Track Error
    he    = obs[7+1]  # Heading Error
    passed = obs[8+1] #　ポイント通過

    # 基本報酬：速度を奨励しつつ、軌道逸脱を罰する

    speed = speed * math.cos(he)
    # いやだけどこうしないとスピード狂がどうにもやめられない
    speed = 20.0 if speed > 20.0 else speed 

    # 追加の報酬修正
    time_bonus_max = 30.0 # 30秒以上なら報酬なし
    rest_time = time_bonus_max - t
    rate = rest_time / time_bonus_max
    rate = 0 if rate < 0 else rate 
    passed_bonus_scale = rate

# 報酬を明確な時だけにする
    reward = 5.0 * passed_bonus_scale if passed else 0
#    reward = speed * delta_time                  # 前向きに進んでるか
#    reward -= 0.1 * abs(cte)                   # 軌道からのずれを罰する
#    reward -= 0.05 * abs(he)                   # 向きのズレも罰する
#    reward += 1 if passed else 0       #1000ポイントあると1000貰える

    # 逆走など明らかに異常な場合に罰則
    if speed < -0.1:
        reward -= 5.0

    # 一周したので残り時間から報酬追加
    if rest_time < 0.0:
        # 30秒以上経過したので失敗
        reward -= 0.01
    elif is_off_track(obs):
        # コースアウトは大きな罰だが、回復の見込みは普通にあるので終了にはしない
        reward -= 0.1  # 罰として明確に伝える

    return reward

def is_off_track(obs, max_perp_error=1.2):
    """
    車両がコースから外れたかを判定する関数

    Parameters:
        obs : np.ndarray
            [x, y, yaw, v, heading_error, perp_error]
        max_perp_error : float
            横ずれ（perpendicular error）の許容上限 [m]

    Returns:
        bool : True なら off track（脱輪）
    """
    perp_error = obs[6+1]  # 横方向の誤差

#  通り過ぎこそ罰にしないと
    if perp_error > max_perp_error:
        return True
    else:
        return False

# ---------------------------------------------------------------------------
# 4) メインループ（別スレッドで回す）
# ---------------------------------------------------------------------------
def run_control_loop(scene, car,sphere,bc_model):
    # — DOF index —
    steer_left  = car.get_joint("fl_steer_joint").dofs_idx_local[0]
    steer_right = car.get_joint("fr_steer_joint").dofs_idx_local[0]
    wheel_rl    = car.get_joint("rl_wheel_joint").dofs_idx_local[0]
    wheel_rr    = car.get_joint("rr_wheel_joint").dofs_idx_local[0]
    idx_steer   = [steer_left, steer_right]
    idx_wheels  = [wheel_rl, wheel_rr]

    # — PID state —
    integ_speed_error = 0.0

    waypoint_direc = -1 if random.randint(0,100) < 50 else 1#1で正方向、-1で逆方向
    start_waypoint_idx = random.randint(0, len(WAYPOINTS)-1)
    end_waypoint_idx = (start_waypoint_idx - waypoint_direc) % len(WAYPOINTS)

    waypoint_idx = start_waypoint_idx

    # 車の初期位置
    set_car_start_pos(car,start_waypoint_idx,waypoint_direc)


    # ----- 初期化フェーズ -----
    latency_filter = LatencyFilter(tau=FILTER_TAU, dt=CONTROL_DT)

    test_counter = 0


    # 教師データ
    data_log = []

    # 経過時間の記録
    t = 0.0

    # トータル報酬を可視化
    reward_total = 0.0

    # Debug arrow
    debug_arrow_segment = None
    debug_arrow_target = None

# ----- 制御ループ（例: Genesis4D の step() 内など） -----
    for step in range(20_0000):

        #新しい制御

        # ───────────
        # 1) 現在位置（車体 root link の COM）
        pos_world = car.get_links_pos()[0]

        # 2D だけ使う
        pos_xy = car.get_dofs_position()[0:2]
        pos_xy = np.array(pos_xy)  # Taichi Vector → numpy

        # 3) ヘディング角（ヨー）取得
        chassis_quat = car.get_links_quat()[0]  # (w, x, y, z)
        qw, qx, qy, qz = chassis_quat
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2*(qy*qy + qz*qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # 4) 速度取得
        car_speed = np.array(car.get_dofs_velocity()).mean()

        # 5) 速度依存ルックアヘッドを計算
        L = compute_lookahead(v=car_speed, k_la=K_LOOK, v_eps=V_EPS)

        # 同じアルゴリズムに合わせないと駄目なので仕方なく
        # これの変動が過敏すぎてターゲット位置をあらぶらせている
        L = 1.25#L if L < 1.5 else 1.5

        # シンプルに次の通過点へ進めるモード
        # チェックポイント通過チェック
        passed,waypoint_idx = check_waypoint_passed(pos_xy, WAYPOINTS, waypoint_idx,waypoint_direc)

        # チェックポイントはそのまま渡す
        target_wp = get_wp_position(waypoint_idx,WAYPOINTS)
        target_next_wp = get_wp_position(waypoint_idx+waypoint_direc,WAYPOINTS)

        # ターゲット球
        if sphere is not None:

            # lookaheadによる動的算出位置
            sphere.set_pos((
                target_wp[0],
                target_wp[1],
                0.5))

            # 現在インデックス位置
#            sphere.set_pos((
#                WAYPOINTS[waypoint_idx][0],
#                WAYPOINTS[waypoint_idx][1],
#                0.5))

            # インデックス全体の位置確認
#            sphere.set_pos((
#                WAYPOINTS[test_counter%len(WAYPOINTS)][0],
#                WAYPOINTS[test_counter%len(WAYPOINTS)][1],
#                0.5))
#            test_counter+=1


        # ターゲット方向
        dx, dy = target_wp - pos_xy
        segment = target_next_wp - target_wp

        # Debug arrow
        if debug_arrow_segment is not None:
            scene.clear_debug_object(debug_arrow_segment)

        segment_len = np.linalg.norm(segment)
        if segment_len > 0.001:
            scale = 0.5 * (1.0 / segment_len)
        else:
            scale = 1.0

        debug_arrow_segment = scene.draw_debug_arrow(
                pos=(target_wp[0], target_wp[1], 0.1),
                vec=(segment[0]*scale, segment[1]*scale, 0.0),
                radius=0.005, color=(0, 0, 1, 0.5))  # Blue
        
        # Debug arrow
        if debug_arrow_target is not None:
            scene.clear_debug_object(debug_arrow_target)

        debug_arrow_target = scene.draw_debug_arrow(
                pos=(pos_xy[0], pos_xy[1], 0.1),
                vec=(dx, dy, 0.0),
                radius=0.005, color=(0, 1, 0, 0.5))  # Green


        # 残り時間カウント
        time_bonus_max = 30.0 # 30秒以上なら報酬なし
        rest_time = time_bonus_max - t

        if is_mode_ai:

            # AIによる自動運転モード

            # AI入力ベクトルを作成して推論

            # ターゲット方向
            dx, dy = target_wp - pos_xy
            target_yaw = math.atan2(dy, dx)

            # セグメント方向
            segment = target_next_wp - target_wp

            # ヘディング誤差（-pi ～ +pi に wrap）
            # ※車体とターゲット方向の角度差
            heading_error = target_yaw - yaw
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

            # CTE（ターゲット方向に直交する方向への距離）
            target_dir = target_wp - pos_xy
            target_dir = target_dir / np.linalg.norm(target_dir)
            segment_dir = segment / np.linalg.norm(segment)
            inner_angle = np.dot(target_dir, segment_dir)
            # 1.0 0.0 -1.0
            # ↓ x 1.0
            # -1.0 0.0 1.0
            # ↓ + 1.0
            # 0.0 1.0 2.0   ※真正面で0.0　真横で1.0 真後ろで2.0
            perp_error = 1.0 - inner_angle

            #まだチェックポイント上に乗っていないのでコースアウトは無視する
            if waypoint_idx == start_waypoint_idx:
                perp_error = 0.0

# ９次元に拡張し順序を合わせる
            input_array = np.array([
                target_wp[0], target_wp[1],
                pos_xy[0], pos_xy[1], yaw, car_speed,
                perp_error,heading_error,passed
            ], dtype=np.float32)
#            input_array = np.array([
#                pos_xy[0], pos_xy[1], yaw, car_speed,
#                target_wp[0], target_wp[1]
#            ], dtype=np.float32)

            input_tensor = torch.tensor(input_array)

            with torch.no_grad():
                steer_angle, throttle = bc_model(input_tensor).tolist()
                # 物理的に許容できる角度に制限
                steer_angle = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer_angle))

        else:
        
            # 6) Pure-Pursuit で生ステア角を計算
            #    WAYPOINTS が (N,2) 形式ならそのまま渡せる
            delta_pp = pure_pursuit_steer(
                target_wp=target_wp,
                pos_xy=pos_xy,
                yaw=yaw,
                lookahead=L,
                wheelbase=1.2,#1.0 0.4 0.3
            )

            # 7) 一階遅れフィルタでステア角を平滑化
            steer_angle = latency_filter(delta_pp)

            # ───────────
            # 8) 車速 PID
            speed_error = TARGET_SPEED - car_speed
            integ_speed_error += speed_error * scene.dt
            throttle = KP_SPEED * speed_error + KI_SPEED * integ_speed_error
            # Clip しておく
            throttle = max(-FORCE_CLIP, min(FORCE_CLIP, throttle))

            # 観測データ
            is_first = True if waypoint_idx==start_waypoint_idx else False

            obs = get_obs(car_pos=pos_xy
                          ,car_vel=car_speed
                          ,car_yaw=yaw
                          ,target_wp=target_wp
                          ,target_next_wp=target_next_wp
                          ,passed=passed
                          ,is_first_check_point=is_first)
            # 報酬
            reward = compute_reward(obs=obs,t=t)

            reward_total += reward

            print(f"[{t:.3f}]教師 reward {reward:.2f} total {reward_total:.2f}")

            # 教師データ記録
            data_log.append({
                # 環境
                "target_wp_x": obs[0],# target_wp[0],# ９次元に拡張し順序を合わせる
                "target_wp_y": obs[1],  #target_wp[1],# ９次元に拡張し順序を合わせる
                "pos_x": obs[2],        #pos_xy[0],
                "pos_y": obs[3],        #pos_xy[1],
                # 断続しないヨー角 １０次元に    
                "yaw_sin": obs[4],          #yaw,
                "yaw_cos": obs[5],          #yaw,
#                "yaw": obs[4],          #yaw,
                "velocity": obs[5+1],     #car_speed,
                "perp_error":obs[6+1],    #0.0,           # ９次元に拡張し順序を合わせる
                "heading_error":obs[7+1], #0.0,        # ９次元に拡張し順序を合わせる
                "passed":obs[8+1],        #0.0,               # ９次元に拡張し順序を合わせる
                # 出力
                "steer_angle": steer_angle,
                "throttle": throttle,       
                # 報酬
                "reward": reward       
            })

        # 一周したらおわり
        if waypoint_idx == end_waypoint_idx or rest_time < 0.0:
            # 教師データとして保存 
            df = pd.DataFrame(data_log)
            df.to_csv("expert_data/expert_data.csv", index=False)
            return


        # 9) Command 発行
        #    ここで steer_angle, throttle を実行関数に渡す
        car.control_dofs_position([steer_angle, steer_angle], idx_steer)
        car.control_dofs_force([throttle, throttle], idx_wheels)

        # 経過時間の記録
        t += scene.dt

        # 10) Genesis4D のタイムステップを回す
        scene.step()

#        # Optional: 可視化確認
#        if step % 200 == 0:
#            print(f"step {step:5d} | pos=({pos_xy[0]: .2f},{pos_xy[1]: .2f}) "
#                f"| car_speed={wheel_vel: .2f} m/s | steer={steer_angle: .2f} rad")

# Viewer ありで CPU が苦しい場合は少し sleep# ---------------------------------------------------------------------------
# 2) Genesis 初期化 & シーン構築
# ---------------------------------------------------------------------------
def build_scene(path_to_mjcf: str | Path):

    gs.init(backend=gs.gpu,logging_level="warning")  # ← CPU でも OK
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0, 0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3*3, 3*3, 2*3),
            camera_lookat=(0, 0, 0),
        ),
        show_viewer=False if is_mode_fast else True,#True,
    )

    # 地面（URDF / MJCF どちらでも）―― fixed=True で質量 0 扱い
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    # 走れる道
    road = scene.add_entity(gs.morphs.Mesh(file="meshes/debug_road/bernoulli_a50_lane025.obj", fixed=True, scale=4.0))

    # simple_car
    car = scene.add_entity(gs.morphs.MJCF(file=str(path_to_mjcf), scale=1.0))

    # 球体を生成（中心：原点、高さ0.5、半径0.5、固定）
    sphere = scene.add_entity(gs.morphs.MJCF(file=str("xml/ant_grasp_ball.xml"), scale=0.1))

    # シーン構築
    scene.build()

    return scene, car ,sphere

        # time.sleep(0.001)


# ---------------------------------------------------------------------------
# 5) エントリポイント
# ---------------------------------------------------------------------------
def main():

    random.seed(time.time())  # 毎回違うシードになる

    scene, car, sphere = build_scene("xml/simple_car.xml")

    # 車の初期位置
    car.set_pos(( -3.8, 0.0, 0.30 ))   # 8 の字左端スタート
    quat = euler_to_quat((0.0, 0.0, 0.0))
    car.set_quat(quat)

    # モデル読み込み
    import os

    bc_model = None

    if os.path.exists("models/bc_model.pth"):
        bc_model = ControlMLP()
        bc_model.load_state_dict(torch.load("models/bc_model.pth"))
        bc_model.eval()




    # 制御ループを別スレッドで
    ctrl_thread = threading.Thread(target=run_control_loop, args=(scene, car,sphere,bc_model), daemon=True)
    ctrl_thread.start()

    # メインスレッドは viewer が閉じられるまでブロック
    ctrl_thread.join()


if __name__ == "__main__":
    main()
