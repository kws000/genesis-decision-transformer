import os
import re
import time

import gym
import numpy as np
from gym import spaces

import genesis as gs

import math
from math import radians

from genesis.utils.geom  import euler_to_quat

from utils.trajectory_utils import yaw_to_sin_cos
from utils.trajectory_utils import sin_cos_to_yaw

import threading
import random
import pyautogui



#最新モデルでリプレイする※作りかけ封印
#REPLAY_MODE = True

# Pure-Pursuit + フィルタ用パラメータ
K_LOOK = 1.0#1.5#0.6#1.2           # ルックアヘッド・タイムスケール [s]
V_EPS = 0.5#0.1            # 最低速度下限 [m/s]
MAX_STEER_RAD = 3.1415926535 * 80.0 / 180    # ステア最大角度

# ビュアーやSleepをスキップする高速モード
is_mode_fast = True#False




class GenesisScene:
    
    def __init__(self):

        gs.init(backend=gs.gpu,logging_level="warning")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(gravity=(0, 0, -9.81)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1, 1, 5),
                camera_lookat=(0, 0, 0),
            ),
            show_viewer=False if is_mode_fast else True,
        )

# 効かない
#        # ビューワーの設定をカスタマイズ
#        viewer_config = gs.ViewerConfig()
#        viewer_config.window_width = 512
#        viewer_config.window_height = 512
#        self.scene.set_viewer(True, viewer_config)

       # 乱数シード
        random.seed(time.time())  # 毎回違うシードになる

        # 車
        self.car = self._load_car()

        # 地面（URDF / MJCF どちらでも）―― fixed=True で質量 0 扱い
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # ８の字道路
        road = self.scene.add_entity(gs.morphs.Mesh(file="meshes/debug_road/bernoulli_a50_lane025.obj", fixed=True, scale=4.0))

        # ８の字経路ポイント
        self.waypoints = self.generate_bernoulli_waypoints(a=2.0)


        # 現在チェックポイントのインデックス

        self.stable_step = 0

#最新モデルでリプレイする※作りかけ封印
#        if REPLAY_MODE:
#            self.stable_step = self.get_latest_stable_step()
#            self.start_waypoint_idx,self.waypoint_direc = self.get_replay_info(self.stable_step)
#        else:
#            self.waypoint_direc = -1 if random.randint(0,100) < 50 else 1#1で正方向、-1で逆方向
#            self.start_waypoint_idx = random.randint(0, len(self.waypoints)-1)
        self.waypoint_direc = -1 if random.randint(0,100) < 50 else 1#1で正方向、-1で逆方向
        self.start_waypoint_idx = random.randint(0, len(self.waypoints)-1)

        self.end_waypoint_idx = (self.start_waypoint_idx - self.waypoint_direc) % len(self.waypoints)
        self.waypoint_idx = self.start_waypoint_idx

#        # 球体を生成（中心：原点、高さ0.5、半径0.5、固定）
        self.sphere = None
#        self.sphere = self.scene.add_entity(gs.morphs.MJCF(file=str("xml/ant_grasp_ball.xml"), scale=0.1))

        # シーン構築
        self.scene.build()

        # Debug arrow
        self.debug_arrow_segment = None
        self.debug_arrow_target = None


        self.lock = threading.Lock()
        self.kick_step = False
        self.kill_myself = False

        # 高速モードではviewを使わない
        if not is_mode_fast:
            self._thread = threading.Thread(target=self._step_loop, daemon=True)
            self._thread.start()

        self.t = 0.0
        self.dt = self.scene.dt

        self.reward_total = 0.0

    # === 安定ステップを自動判定 ===

    #最新モデルでリプレイする※作りかけ封印
    def get_latest_stable_step(self):
        step_files = [f for f in os.listdir("checkpoints_dir") if re.match(r"step(\d+)\.pt", f)]

        step_ids = []
        for f in step_files:
            match = re.match(r"step(\d+)\.pt", f)
            if match:
                step_ids.append(int(match.group(1)))

        return max(step_ids) if step_ids else -1
    
    # === リプレイ情報取得ヘルパー ===

    #最新モデルでリプレイする※作りかけ封印
    def get_replay_info(self,stable_step):
        replay_path = f"checkpoints/step{stable_step}_replay.txt"
        try:
            with open(replay_path, "r") as f:
                return int(f.readline().strip()),int(f.readline().strip())
        except Exception as e:
            print(f"⚠️ リプレイ情報読み込み失敗: {e}")
            return -int("0"),-int("0")

    # --共通のメソッド(vehicle_control_drlにもある)--

    # selfつけるだけ、ダルすぎる、、どうまとめよう

    def generate_bernoulli_waypoints(
            self,
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

    def get_wp_position(self,wp_idx: int,waypoints: np.ndarray):

        wp_idx = wp_idx % len(waypoints)

        return waypoints[wp_idx][:2]

    # vは正規化不要とのこと
    #roll pitch yaw の順は ROS では標準
    def vector_to_euler(self,v):
        x, y, z = v
        yaw = np.arctan2(y, x)
        pitch = np.arctan2(-z, np.sqrt(x**2 + y**2))
        roll = 0  # ロールは定義できない（方向ベクトルだけでは不定）
        return np.degrees([roll, pitch, yaw])

    def set_car_start_pos(self,car,waypoint_idx: int,waypoint_direc: int):

        # 車の初期位置
        target_wp = self.get_wp_position(waypoint_idx,self.waypoints)
        target_next_wp = self.get_wp_position(waypoint_idx+waypoint_direc,self.waypoints)

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
            angle = self.vector_to_euler((segment_vec[0],segment_vec[1],0.0))
            quat = euler_to_quat(angle)
            car.set_quat(quat)
        else:
            car.set_pos(( -3.8, 0.0, 0.30 ))   # 8 の字左端スタート
            quat = euler_to_quat((0.0, 0.0, 0.0))
            car.set_quat(quat)


    def compute_lookahead(self,
                          v: float,
                        k_la: float = 1.0,
                        v_eps: float = 0.1) -> float:
        """
        速度依存ルックアヘッド距離 L を返す
        - v     : 現在速度 [m/s]
        - k_la  : ルックアヘッド・タイムスケール [s]
        - v_eps : 最低速度下限 [m/s]
        """
        return k_la * max(v, v_eps)

    def check_waypoint_passed(self,pos: np.ndarray, waypoints: np.ndarray, current_wp_idx: int,waypoint_direc: int, threshold: float = 1.0):
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

        # ラップ    
        current_wp_idx = current_wp_idx % len(waypoints)

        wp = waypoints[current_wp_idx]
        distance = np.linalg.norm(pos - wp[:2])

        if distance < threshold:
            return True, current_wp_idx + waypoint_direc  # 通過とみなして次へ
        else:
            return False, current_wp_idx

    def _load_car(self):
        return self.scene.add_entity(
            gs.morphs.MJCF(file=str("xml/simple_car.xml"), scale=1.0),
        )


    def _step_loop(self):

        # 指示があった時だけシミュレーションする
        while True:
            
            with self.lock:

                if self.kill_myself:
                    break

                if self.kick_step:
                    self.scene.step()
                    self.kick_step = False
            
            time.sleep(0.01)

    def reset(self):

        # — DOF index —
        steer_left  = self.car.get_joint("fl_steer_joint").dofs_idx_local[0]
        steer_right = self.car.get_joint("fr_steer_joint").dofs_idx_local[0]
        wheel_rl    = self.car.get_joint("rl_wheel_joint").dofs_idx_local[0]
        wheel_rr    = self.car.get_joint("rr_wheel_joint").dofs_idx_local[0]
        idx_steer   = [steer_left, steer_right]
        idx_wheels  = [wheel_rl, wheel_rr]

        with self.lock:
            self.scene.reset()
            self.car.set_pos(( -3.8, 0.0, 0.30 ))   # 8 の字左端スタート
            quat = euler_to_quat((0.0, 0.0, 0.0))
            self.car.set_quat(quat)

            self.car.control_dofs_position([0.0]*2, idx_steer)
            self.car.control_dofs_force([0.0]*2, idx_wheels)

            self.t = 0.0

        # 開始チェックポイントの決定
        self.waypoint_direc = -1 if random.randint(0,100) < 50 else 1#1で正方向、-1で逆方向
        self.start_waypoint_idx = random.randint(0, len(self.waypoints)-1)
        self.end_waypoint_idx = (self.start_waypoint_idx - self.waypoint_direc) % len(self.waypoints)
        self.waypoint_idx = self.start_waypoint_idx

        # 車の初期位置
        self.set_car_start_pos(self.car,self.start_waypoint_idx,self.waypoint_direc)

        # 報酬リセット
        self.reward_total = 0.0

#        # 最初からワイヤーにしたい
#        if self.scene.viewer is not None:
#            time.sleep(1)  # Viewerが起動するまで待つ
#            pyautogui.press('d')  # ワイヤーフレームトグル

        return self._get_obs()

    # 強化学習ライブラリ側から呼ばれる
    def step(self, steer, throttle):

        # — DOF index —
        steer_left  = self.car.get_joint("fl_steer_joint").dofs_idx_local[0]
        steer_right = self.car.get_joint("fr_steer_joint").dofs_idx_local[0]
        wheel_rl    = self.car.get_joint("rl_wheel_joint").dofs_idx_local[0]
        wheel_rr    = self.car.get_joint("rr_wheel_joint").dofs_idx_local[0]
        idx_steer   = [steer_left, steer_right]
        idx_wheels  = [wheel_rl, wheel_rr]

        # 物理的に許容できる角度に制限
        steer = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer))

        # コマンド実行
        self.car.control_dofs_position([steer]*2, idx_steer)
        self.car.control_dofs_force([throttle]*2, idx_wheels)

        # 高速モードではviewを使わない
        if is_mode_fast:
            # 高速モード
            self.scene.step()
        else:
            # シミュレーションスレッドと排他制御
            with self.lock:
                self.kick_step = True

            # viewerスレッド側でscene.step()されるのを待つ
            while(self.kick_step):
                # シミュレーションが実行されるまで待つ
                time.sleep(0.01)

        # ステップ実行後
        obs = self._get_obs()
        reward = self._compute_reward(obs,self.t)

        # 制限時間かコースアウトなら学習終了
        done = False

        if self.waypoint_idx == self.end_waypoint_idx:
# チェックポイントと残り時間で報酬を細分化する            
#            # 時間内に到着したので成功報酬
#            reward += rest_time * 100
            done = True
#        elif self.is_off_track(obs):
#            reward -= 1.0  # 罰として明確に伝える
#            #回復の見込みが普通にあるので終了niacinamide
        elif self.t > 60.0:
            # 時間かかりすぎ終了
            reward -= 1000
            done = True            
        elif self.t > 30.0 and self.waypoint_idx == self.start_waypoint_idx:
            # 十分な時間が立ったのにまだ最初のチェックポイントを通過していない
            reward -= 1000
            done = True
        elif self.reward_total < -250:
            # 大きく損失していてもう回復が見込みめない
            done = True

        self.reward_total += reward

        print(f"[{self.t:.3f}]生徒 reward {reward:.2f} total {self.reward_total:.2f}")

        self.t += self.dt

        return obs, reward, done, {}

#  通り過ぎこそ罰にしないと
    def is_off_track(self, obs, max_perp_error=1.2):
#    def is_off_track(self, obs, max_perp_error=2.0):
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

    def _get_obs(self):

        # — DOF index —
        steer_left  = self.car.get_joint("fl_steer_joint").dofs_idx_local[0]
        steer_right = self.car.get_joint("fr_steer_joint").dofs_idx_local[0]
        wheel_rl    = self.car.get_joint("rl_wheel_joint").dofs_idx_local[0]
        wheel_rr    = self.car.get_joint("rr_wheel_joint").dofs_idx_local[0]
        idx_steer   = [steer_left, steer_right]
        idx_wheels  = [wheel_rl, wheel_rr]

        pos = np.array(self.car.get_dofs_position())[:2]
        vel = np.array(self.car.get_dofs_velocity()).mean()

        quat = self.car.get_links_quat()[0]  # chassisの回転（w, x, y, z）
        siny_cosp = 2 * (quat[0]*quat[3] + quat[1]*quat[2])
        cosy_cosp = 1 - 2 * (quat[2]**2 + quat[3]**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # 5) 速度存ルックアヘッドを計算
        L = self.compute_lookahead(v=(float)(vel), k_la=K_LOOK, v_eps=V_EPS)

        # これの変動が過敏すぎてターゲット位置をあらぶらせている
        L = 1.25#L if L < 1.5 else 1.5

        # シンプルに次の通過点へ進めるモード
        # チェックポイント通過チェック

        passed,self.waypoint_idx = self.check_waypoint_passed(pos, self.waypoints, self.waypoint_idx,self.waypoint_direc)
        # チェックポイントはそのまま渡す
        target_wp = self.get_wp_position(self.waypoint_idx,self.waypoints)
        target_next_wp = self.get_wp_position(self.waypoint_idx+self.waypoint_direc,self.waypoints)

        # ターゲット方向
        dx, dy = target_wp - pos
        target_yaw = math.atan2(dy, dx)

        # セグメント方向
        segment = target_next_wp - target_wp


        # ターゲット球
        if self.sphere is not None:

            # lookaheadによる動的算出位置
            self.sphere.set_pos((
                target_wp[0],
                target_wp[1],
                0.5))

        # Debug arrow
        if self.debug_arrow_segment is not None:
            self.scene.clear_debug_object(self.debug_arrow_segment)

        segment_len = np.linalg.norm(segment)
        if segment_len > 0.001:
            scale = 0.5 * (1.0 / segment_len)
        else:
            scale = 1.0

        self.debug_arrow_segment = self.scene.draw_debug_arrow(
                pos=(target_wp[0], target_wp[1], 0.1),
                vec=(segment[0]*scale, segment[1]*scale, 0.0),
                radius=0.005, color=(0, 0, 1, 0.5))  # Blue
        
        # Debug arrow
        if self.debug_arrow_target is not None:
            self.scene.clear_debug_object(self.debug_arrow_target)

        self.debug_arrow_target = self.scene.draw_debug_arrow(
                pos=(pos[0], pos[1], 0.1),
                vec=(dx, dy, 0.0),
                radius=0.005, color=(0, 1, 0, 0.5))  # Green


        # ヘディング誤差（-pi ～ +pi に wrap）
        # ※車体とターゲット方向の角度差
        heading_error = target_yaw - yaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # CTE（ターゲット方向に直交する方向への距離）
        # ※道路と

        # wp0 = current segment start, wp1 = segment end

        # perp_dir 計算間違ってる？
        is_perp_bugfix = True

        if is_perp_bugfix:
            target_dir = target_wp - pos
            target_dir = target_dir / np.linalg.norm(target_dir)
            segment_dir = segment / np.linalg.norm(segment)
            inner_angle = np.dot(target_dir, segment_dir)
            # 1.0 0.0 -1.0
            # ↓ x 1.0
            # -1.0 0.0 1.0
            # ↓ + 1.0
            # 0.0 1.0 2.0   ※真正面で0.0　真横で1.0 真後ろで2.0
            perp_error = 1.0 - inner_angle
        else:
            rel_pos = pos - target_wp
            # segment に直交するベクトル
            perp_dir = np.array([-segment[1], segment[0]])  # [-dy, dx]
            perp_dir = perp_dir / np.linalg.norm(perp_dir)
            perp_error = np.dot(rel_pos, perp_dir)

        #まだチェックポイント上に乗っていないのでコースアウトは無視する
        if self.waypoint_idx == self.start_waypoint_idx:
            perp_error = 0.0

# 断続しないヨー角 １０次元に
        car_yaw_sin,car_yaw_cos = yaw_to_sin_cos(yaw)
        return np.array([target_wp[0], target_wp[1], pos[0], pos[1], car_yaw_sin,car_yaw_cos, vel, perp_error, heading_error,passed], dtype=np.float32)
#        # ９次元に拡張し順序を合わせる　要素が間違っている
#        return np.array([target_wp[0], target_wp[1], pos[0], pos[1], yaw, vel, perp_error, heading_error,passed], dtype=np.float32)

    def _compute_reward(self, obs, t):
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
#        reward = speed * self.scene.dt                            # 前向きに進んでるか
#        reward -= 0.1 * abs(cte)                   # 軌道からのずれを罰する
#        reward -= 0.05 * abs(he)                   # 向きのズレも罰する
#        reward += 1 if passed else 0

        # 逆走など明らかに異常な場合に罰則
        if speed < -0.1:
            reward -= 1.0

        # 一周したので残り時間から報酬追加
        if rest_time < 0.0:
            # 30秒以上経過したので失敗
            reward -= 0.01
        elif self.is_off_track(obs):
            # コースアウトは大きな罰だが、回復の見込みは普通にあるので終了にはしない
            reward -= 0.1  # 罰として明確に伝える

        return reward

    def close(self):
        with self.lock:
            self.kill_myself = True
        self._thread.join()


class GenesisEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),
                                       high=np.array([1.0, 1.0]), dtype=np.float32)

        # _get_obs の要素を増やしたら、shapeの数を増やす必要がある
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.scene = GenesisScene()

    def reset(self):
        return self.scene.reset()

    def step(self, action):
        steer, throttle = action
        return self.scene.step(steer, throttle)

    def close(self):
        self.scene.close()

