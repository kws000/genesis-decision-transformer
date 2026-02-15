
import math
from math import radians
import threading
import time
from pathlib import Path
import numpy as np
import pkgutil, inspect
import genesis as gs # type: ignore
from genesis.utils.geom  import euler_to_quat # type: ignore
import pandas as pd
import random
import torch
from utils.trajectory_utils import yaw_to_sin_cos
from utils.trajectory_utils import sin_cos_to_yaw

#ボトルネック認識とVmax魂の注入 3.1 
from pre_training.laws.laws_vmax_infer import VmaxFactorized  # 既存のラッパ


#ボトルネック認識とVmax魂の注入 3.1 obsビルダー
from geo_utils import curvature_from_wps

#ボトルネック認識とVmax魂の注入 3.1 obsビルダー
OBS_V2_KEYS = [
    # 既存10
    "target_wp_relative_x","target_wp_relative_y","pos_x","pos_y",
    "yaw_sin","yaw_cos","velocity","perp_error","heading_error","passed",
    # 追加9（VMAX塊）
    "kappa_local","mu_local","vmax_local","v_ratio","headroom",
    "vmax_min_hH","vmax_mean_hH","vmax_slope_hH","limit_v_target",
]

def build_obs_v2_pure(pos_xy, yaw, vel, passed,
                      target_wp, target_next_wp,
                      waypoint_idx, waypoint_direc, WAYPOINTS,
                      vmax_model: VmaxFactorized,
                      H_preview=10, mu_default=0.8, speed_limit=None):
    """
    Pure-Pursuit用に、Envと独立に OBS_V2(19次元) を生成
    """
    # ego座標でのターゲット相対位置
    dx, dy = target_wp - pos_xy
    target_yaw = math.atan2(dy, dx)

    # セグメント方向
    segment = target_next_wp - target_wp
    # ヘディング誤差（-pi..pi）
    heading_error = (target_yaw - yaw + np.pi) % (2*np.pi) - np.pi

    # CTE 近似（方向ベクトル内積から）
    tdir = (target_wp - pos_xy); tdir = tdir / (np.linalg.norm(tdir) + 1e-9)
    sdir = segment / (np.linalg.norm(segment) + 1e-9)
    perp_error = 1.0 - float(np.dot(tdir, sdir))
    # 初手だけ無視する等のルールは上位で適用可

    # ego変換
    target_wp_relative_x =  math.cos(yaw)*dx + math.sin(yaw)*dy
    target_wp_relative_y = -math.sin(yaw)*dx + math.cos(yaw)*dy

    yaw_sin, yaw_cos = math.sin(yaw), math.cos(yaw)


    assert H_preview >= 1, f"H_preview must be >=1, got {H_preview}"
    assert WAYPOINTS is not None and len(WAYPOINTS) >= 3, f"WAYPOINTS too short: {len(WAYPOINTS) if WAYPOINTS is not None else None}"
    assert isinstance(waypoint_idx, int) and 0 <= waypoint_idx < len(WAYPOINTS), f"bad waypoint_idx={waypoint_idx}"
    assert waypoint_direc in (-1, 1), f"bad waypoint_direc={waypoint_direc}"

    # 先読み κ と vmax
    kappas_H, ds_H = curvature_from_wps(WAYPOINTS, waypoint_idx, waypoint_direc, H_preview)
    mus_H = [mu_default]*len(kappas_H)
    mu_local = mus_H[0] if mus_H else mu_default
    kappa_local = float(kappas_H[0]) if kappas_H else 0.0

    vmax_local = float(vmax_model.from_kappa(kappa_local, mu_local))
    vmax_preview = vmax_model.batch_kappa(kappas_H, mus_H) if len(kappas_H) else np.array([vmax_local], np.float32)

    vmax_min_hH   = float(np.min(vmax_preview))
    vmax_mean_hH  = float(np.mean(vmax_preview))
    vmax_slope_hH = float((vmax_preview[-1] - vmax_preview[0]) / max(1, len(vmax_preview)-1)) if len(vmax_preview)>=2 else 0.0

    v_ratio  = float(vel) / (vmax_local + 1e-3)
    headroom = vmax_local - float(vel)

    limit_v_target = float(min(speed_limit if speed_limit is not None else float("inf"), vmax_min_hH))

    obs = np.array([
        target_wp_relative_x, target_wp_relative_y, float(pos_xy[0]), float(pos_xy[1]),
        yaw_sin, yaw_cos, float(vel), float(perp_error), float(heading_error), float(passed),
        float(kappa_local), float(mu_local), float(vmax_local), float(v_ratio), float(headroom),
        float(vmax_min_hH), float(vmax_mean_hH), float(vmax_slope_hH), float(limit_v_target)
    ], dtype=np.float32)
    return obs


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

# システム

# False:PurePursaitによる運転と教師CSVの収集、True:bc_modelによる推論運転
is_mode_bc_model = False#True
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

#計画と行動のマルチタスクモデル　教師データに計画を入れる
# ===== Plan A（短ホライズン参照点）設定 =====
PLAN_M = 3
# 速度依存ルックアヘッド LA(v) に対する比率（turn-in / apex / track-out のイメージ）
PLAN_FACTORS = [0.7, 1.4, 2.2]  # len=PLAN_M
PLAN_LA_MIN = 0.5#5.0   # [m]
PLAN_LA_MAX = 3.0#30.0  # [m]


# vは正規化不要とのこと
#roll pitch yaw の順は ROS では標準
def vector_to_euler(v):
    x, y, z = v
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(-z, np.sqrt(x**2 + y**2))
    roll = 0  # ロールは定義できない（方向ベクトルだけでは不定）
    return [roll, pitch, yaw]
#    return np.degrees([roll, pitch, yaw])

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

#計画と行動のマルチタスクモデル　教師データに計画を入れる
# === 追加: ワールド→車体座標 ===
def world_to_ego(px: float, py: float, ego_pos: np.ndarray, ego_yaw: float) -> tuple[float, float]:
    dx = px - ego_pos[0]
    dy = py - ego_pos[1]
    c = math.cos(ego_yaw); s = math.sin(ego_yaw)
    # ローカルx=前方, y=左+（PPの式と整合）
    ex =  c * dx + s * dy
    ey = -s * dx + c * dy
    return float(ex), float(ey)

#計画と行動のマルチタスクモデル　教師データに計画を入れる
# === 追加: 折れ線WAYPOINTS上を距離dだけ先に進んだ点（世界座標）を返す ===
def point_ahead_on_waypoints(waypoints: np.ndarray,
                             start_pos_world: np.ndarray,
                             start_wp_idx: int,
                             direc: int,
                             distance: float) -> np.ndarray:
    """
+    折れ線 path 上の「現在位置→次WP→…」に沿って distance[m] 先の点を線形補間で返す。
+    - start_pos_world: 現在の車体位置 [x,y]
+    - start_wp_idx   : 現在の「ターゲットWP」のインデックス（pathの参照起点）
+    - direc          : +1 or -1
+    """

#pos_xy→xyz
    wp = np.asarray(waypoints)
    # (N,3) の場合は XY だけ使う。 (N,2) はそのまま
    if wp.ndim != 2 or wp.shape[0] == 0:
        return np.asarray(start_pos_world[:2], dtype=np.float32)
    if wp.shape[1] >= 2:
        wp_xy = wp[:, :2].astype(np.float32)
    else:
        # まさかの1列なら0で拡張
        wp_xy = np.pad(wp.astype(np.float32), ((0,0),(0,2-wp.shape[1])), mode="constant")
    N = wp_xy.shape[0]
    # 1本目の区間: 現在位置(xy) -> target_wp(xy)
    seg_start = np.asarray(start_pos_world[:2], dtype=np.float32)
    seg_end   = wp_xy[start_wp_idx % N]
#    N = len(waypoints)
#    # 1本目の区間: 現在位置 -> target_wp
#    seg_start = start_pos_world.astype(np.float32)
#    seg_end   = waypoints[start_wp_idx % N].astype(np.float32)

    remain = float(distance)
    # まずは現在位置から target_wp まで
    for _ in range(N * 2):  # 安全上限
        v = seg_end - seg_start
        seg_len = float(np.linalg.norm(v))
        if seg_len < 1e-6:
            # 極短セグメントはスキップ
            seg_start = seg_end
        else:
            if remain <= seg_len:
                r = remain / seg_len
#pos_xy→xyz
                pt = seg_start + r * v
                return np.asarray(pt, dtype=np.float32)                
#                return seg_start + r * v
            
            remain -= seg_len
            seg_start = seg_end
        # 以降の区間: wp[i] -> wp[i+direc]
#pos_xy→xyz
        next_idx = (start_wp_idx + direc) % N
        seg_end = wp_xy[next_idx]
#        next_idx = (start_wp_idx + direc) % N
#        seg_end = waypoints[next_idx].astype(np.float32)

        start_wp_idx = next_idx
    # 走り切っても足りない場合は最後の点を返す
#pos_xy→xyz
    return np.asarray(seg_end, dtype=np.float32)
#    return seg_end

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

    #計画と行動のマルチタスクモデル 計画が相対なのでターゲット位置も相対に変更
    target_wp_subvec = target_wp - car_pos
    target_wp_relative_x =  math.cos(car_yaw)*target_wp_subvec[0] + math.sin(car_yaw)*target_wp_subvec[1]
    target_wp_relative_y = -math.sin(car_yaw)*target_wp_subvec[0] + math.cos(car_yaw)*target_wp_subvec[1]
    
    car_yaw_sin,car_yaw_cos = yaw_to_sin_cos(car_yaw)

#計画と行動のマルチタスクモデル 計画が相対なのでターゲット位置も相対に変更
    return np.array([target_wp_relative_x, target_wp_relative_y, car_pos[0], car_pos[1], car_yaw_sin,car_yaw_cos, car_vel, perp_error, heading_error,passed], dtype=np.float32)
#    return np.array([target_wp[0], target_wp[1], car_pos[0], car_pos[1], car_yaw_sin,car_yaw_cos, car_vel, perp_error, heading_error,passed], dtype=np.float32)

# 前に進まない学習を報酬で改善
def compute_reward_teacher(obs, t, stuck_count):
    vel    = float(obs[6])
    he     = float(obs[8])
    passed = float(obs[9]) > 0.5
    vlim   = float(obs[18])

    # (1) 時間コスト
    r_time = -0.003

    # (2) 進捗
    progress = vel * math.cos(he)
    r_prog = 0.02 * progress

    # (3) stuckペナ（1秒以上停止）
    r_stuck = -0.2 if stuck_count > 100 else 0.0

    # passedボーナス（既存思想を残すなら）
    time_bonus_max = 30.0
    rate = max((time_bonus_max - t) / time_bonus_max, 0.0)
    r_pass = (5.0 * rate) if passed else 0.0

    # vmax超過のみ罰
    over = max(vel - vlim, 0.0)
    r_vmax = -0.02 * (over ** 2)

    return r_time + r_prog + r_stuck + r_pass + r_vmax

#def compute_reward(obs,t):
#    # obs = [x, y, yaw, speed, cross_track_err, heading_err]
#    speed = obs[5+1]
#    cte   = obs[6+1]  # Cross Track Error
#    he    = obs[7+1]  # Heading Error
#    passed = obs[8+1] #　ポイント通過
#    # 基本報酬：速度を奨励しつつ、軌道逸脱を罰する
#    speed = speed * math.cos(he)
#    # 追加の報酬修正
#    time_bonus_max = 30.0 # 30秒以上なら報酬なし
#    rest_time = time_bonus_max - t
#    rate = rest_time / time_bonus_max
#    rate = 0 if rate < 0 else rate 
#    passed_bonus_scale = rate
#    reward = 5.0 * passed_bonus_scale if passed else 0
#    # 逆走など明らかに異常な場合に罰則
#    if speed < -0.1:
#        reward -= 5.0
#    # 一周したので残り時間から報酬追加
#    if rest_time < 0.0:
#        # 30秒以上経過したので失敗
#        reward -= 0.01
#    elif is_off_track(obs):
#        # コースアウトは大きな罰だが、回復の見込みは普通にあるので終了にはしない
#        reward -= 0.1  # 罰として明確に伝える
#    return reward

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

    # 教師データ
    data_log = []
    # 経過時間の記録
    t = 0.0
    # トータル報酬を可視化
    reward_total = 0.0
    # Debug arrow
    debug_arrow_segment = None
    debug_arrow_target = None
    debug_arrow_plans = [None,None,None]

    #ボトルネック認識とVmax魂の注入 3.2 
    vmax_model = VmaxFactorized(
        model_path="pre_training/models/vmax_factorized.pt",
        scaler_path="pre_training/models/scaler_factorized.npz",
        g=9.80665, safety=0.85, r_clip=1000.0, device="cpu"
    )
    H_PREVIEW = 10
    MU_DEFAULT = 0.8
    SPEED_LIMIT = None  # 実装あれば数値を入れて min 取る

    # 前に進まない学習を報酬で改善
    stuck_count = 0 

# ----- 制御ループ（例: Genesis4D の step() 内など） -----
    for step in range(20_0000):

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

        # ターゲット方向
        dx, dy = target_wp - pos_xy
        segment = target_next_wp - target_wp

        # Debug arrow
        if debug_arrow_segment is not None:
            scene.clear_debug_object(debug_arrow_segment)

        # Debug arrow
        for arrow_plan in debug_arrow_plans:
            if arrow_plan is not None:
                scene.clear_debug_object(arrow_plan)

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

        if is_mode_bc_model:
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

            input_array = np.array([
                target_wp[0], target_wp[1],
                pos_xy[0], pos_xy[1], yaw, car_speed,
                perp_error,heading_error,passed
            ], dtype=np.float32)

            input_tensor = torch.tensor(input_array)

            with torch.no_grad():
                steer_angle, throttle = bc_model(input_tensor).tolist()
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

#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる

            # 目標速度（vmax と連携するなら min(TARGET_SPEED_BASE, limit_v_target) を使う）
            v_ref = TARGET_SPEED
            # PI（アンチワインドアップ付き）
            speed_error = v_ref - car_speed
            # 飽和前のコントロール
            u_raw = KP_SPEED * speed_error + KI_SPEED * integ_speed_error
            # 物理域で飽和（前進のみ＝非負）
            u_sat = max(0.0, min(FORCE_CLIP, u_raw))
            # ★アンチワインドアップ（条件付き積分：飽和時は誤差が飽和方向に押しているときだけ積分）
            allow_integrate = (
                (0.0 < u_raw < FORCE_CLIP) or
                (u_raw >= FORCE_CLIP and speed_error < 0) or
                (u_raw <= 0.0      and speed_error > 0)
            )
            if allow_integrate:
                integ_speed_error += speed_error * scene.dt
            throttle = u_sat
            # 発進補助（停止近傍で最小トルク）
            if abs(car_speed) < 0.1 and throttle < 0.05 * FORCE_CLIP:
                throttle = 0.05 * FORCE_CLIP
#            speed_error = TARGET_SPEED - car_speed
#            integ_speed_error += speed_error * scene.dt
#            throttle = KP_SPEED * speed_error + KI_SPEED * integ_speed_error
#            # Clip しておく
#            throttle = max(-FORCE_CLIP, min(FORCE_CLIP, throttle))

            # 観測データ
            is_first = True if waypoint_idx==start_waypoint_idx else False

#ボトルネック認識とVmax魂の注入 3.2 
            obs = build_obs_v2_pure(
                pos_xy=pos_xy, yaw=yaw, vel=car_speed, passed=passed,
                target_wp=target_wp, target_next_wp=target_next_wp,
                waypoint_idx=waypoint_idx, waypoint_direc=waypoint_direc,
                WAYPOINTS=WAYPOINTS,
                vmax_model=vmax_model,
                H_preview=H_PREVIEW, mu_default=MU_DEFAULT, speed_limit=SPEED_LIMIT
            )
#            obs = get_obs(car_pos=pos_xy
#                          ,car_vel=car_speed
#                          ,car_yaw=yaw
#                          ,target_wp=target_wp
#                          ,target_next_wp=target_next_wp
#                          ,passed=passed
#                          ,is_first_check_point=is_first)


            #前に進まない学習を報酬で改善 obsを作った直後に、進行度合いのカウンターを進める
            vel = float(obs[6])          # velocity
            he  = float(obs[8])          # heading_error
            progress = vel * math.cos(he)
            # しきい値は箱庭スケールに合わせて後で調整（まずは小さめ）
            if abs(progress) < 0.03:
                stuck_count += 1
            else:
                stuck_count = 0

            # 報酬
#前に進まない学習を報酬で改善
            reward = compute_reward_teacher(obs=obs,t=t,stuck_count=stuck_count)
#            reward = compute_reward_teacher(obs=obs,t=t)
            reward_total += reward

            print(f"[{t:.3f}]教師 reward {reward:.2f} total {reward_total:.2f}")

            # 教師データ記録

            #計画と行動のマルチタスクモデル　教師データに計画を入れる
            # ==== Plan A: 将来参照点（ego座標）を生成 ====
            # 速度連動 lookahead（計画用は固定化せず v に追従）
            LA = L
#            LA = compute_lookahead(v=car_speed, k_la=K_LOOK, v_eps=V_EPS)
#            LA = max(PLAN_LA_MIN, min(PLAN_LA_MAX, LA))
            dists = [f * LA for f in PLAN_FACTORS[:PLAN_M]]
            plan_xy: list[float] = []
            #　計画ベクトル列の可視化
            debug_plan_xy_world: list[float] = []
            for d in dists:
                w_pt = point_ahead_on_waypoints(WAYPOINTS, pos_xy, waypoint_idx, waypoint_direc, d)
                ex, ey = world_to_ego(w_pt[0], w_pt[1], pos_xy, yaw)
                plan_xy.extend([ex, ey])  # [x1,y1,x2,y2,x3,y3]
                #　計画ベクトル列の可視化
                debug_plan_xy_world.extend([w_pt[0],w_pt[1]])

            #　計画ベクトル列の可視化
            for i in range(0,2):
                allow_plan_src_x = debug_plan_xy_world[i*2+0]
                allow_plan_src_y = debug_plan_xy_world[i*2+1]

                allow_plan_dst_x = debug_plan_xy_world[(i+1)*2+0]
                allow_plan_dst_y = debug_plan_xy_world[(i+1)*2+1]

                segment_plan_x = allow_plan_dst_x - allow_plan_src_x
                segment_plan_y = allow_plan_dst_y - allow_plan_src_y

                debug_arrow_plans[i] = scene.draw_debug_arrow(
                        pos=(allow_plan_src_x, allow_plan_src_y, 0.1),
                        vec=(segment_plan_x, segment_plan_y, 0.0),
                        radius=0.005, color=(0, 1, 1, 0.5))  # LightBlue


#ボトルネック認識とVmax魂の注入 3.3 記録
            # ★ 教師データ記録（OBS_V2 固定順＋出力＋報酬＋計画）
            row = dict(zip(OBS_V2_KEYS, obs.tolist()))
            row.update({
                "steer_angle": float(steer_angle),
                "throttle": float(throttle),
                "reward": float(reward),
                "reward_total": float(reward_total),
                "plan_x1": float(plan_xy[0]), "plan_y1": float(plan_xy[1]),
                "plan_x2": float(plan_xy[2]), "plan_y2": float(plan_xy[3]),
                "plan_x3": float(plan_xy[4]), "plan_y3": float(plan_xy[5]),
            })
            data_log.append(row)
#            # 教師データ記録（plan_x*, plan_y* を追加）
#            data_log.append({
#                # 環境
#                "target_wp_x": obs[0],# target_wp[0],# ９次元に拡張し順序を合わせる
#                "target_wp_y": obs[1],  #target_wp[1],# ９次元に拡張し順序を合わせる
#                "pos_x": obs[2],        #pos_xy[0],
#                "pos_y": obs[3],        #pos_xy[1],
#                # 断続しないヨー角 １０次元に    
#                "yaw_sin": obs[4],          #yaw,
#                "yaw_cos": obs[5],          #yaw,
#                "velocity": obs[5+1],     #car_speed,
#                "perp_error":obs[6+1],    #0.0,           # ９次元に拡張し順序を合わせる
#                "heading_error":obs[7+1], #0.0,        # ９次元に拡張し順序を合わせる
#                "passed":obs[8+1],        #0.0,               # ９次元に拡張し順序を合わせる
#                # 出力
#                "steer_angle": steer_angle,
#                "throttle": throttle,       
#                # 報酬
##計画と行動のマルチタスクモデル　教師データに計画を入れる
#                "reward": reward,       # ← ステップ報酬（学習に使う）
#                "reward_total": reward,  # ← 参考：累積（解析用）                
#                # 計画（ego座標の将来点; M=3 固定）
#                "plan_x1": plan_xy[0], "plan_y1": plan_xy[1],
#                "plan_x2": plan_xy[2], "plan_y2": plan_xy[3],
#                "plan_x3": plan_xy[4], "plan_y3": plan_xy[5],                
##                "reward": reward       
#            })

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

#        print(f"教師の control_dofs_force throttle={throttle}")


        # 経過時間の記録
        t += scene.dt
        # 10) Genesis4D のタイムステップを回す
        scene.step()


# Viewer ありで CPU が苦しい場合は少し sleep# ---------------------------------------------------------------------------
# 2) Genesis 初期化 & シーン構築
# ---------------------------------------------------------------------------
def build_scene(path_to_mjcf: str | Path):

#急にエラーで動かなくなった、、、
#   Backend tkagg is interactive backend. Turning interactive mode on.
#   [Genesis] [01:24:48] [WARNING] No Intel XPU device available. Falling back to CPU for torch device.
#   Assertion failed: pCreateInfo->vulkanApiVersion == 0 || (((uint32_t)(pCreateInfo->vulkanApiVersion) >> 22U) == 1 && (((uint32_t)(pCreateInfo->vulkanApiVersion) >> 12U) & 0x3FFU) <= 3), file C:\Users\buildbot\actions-runner\_work\taichi\taichi\external\VulkanMemoryAllocator\include\vk_mem_alloc.h, line 16039
    gs.init(backend=gs.cpu,logging_level="warning")
#gs.init(backend=gs.gpu,logging_level="warning")  # ← CPU でも OK
    
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
