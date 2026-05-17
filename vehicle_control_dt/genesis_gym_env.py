import os
import re
import time

import gym
import numpy as np
from gym import spaces

import config

import genesis as gs # type: ignore
from genesis.utils.geom  import euler_to_quat # type: ignore

import math
from math import radians


from utils.trajectory_utils import yaw_to_sin_cos
from utils.trajectory_utils import sin_cos_to_yaw

import threading
import random
import pyautogui



#計画と行動のマルチタスクモデル
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict
import numpy as np

#計画と行動のマルチタスクモデル
from typing import Optional, Tuple

#ボトルネック認識とVmax魂の注入
from pre_training.laws.laws_vmax_infer import VmaxFactorized

#ボトルネック認識とVmax魂の注入 3.1 obsビルダー
# genesis_gym_env.py
from geo_utils import curvature_from_wps


## プロンプト生成で町を作る
#from genesis import generate_scene_from_prompt
#from genesis.scene import Scene
#from genesis.viewer import Viewer

# プロンプト生成で町を作る
PROMPT_MODE = False#True

#最新モデルでリプレイする※作りかけ封印
#REPLAY_MODE = True

# Pure-Pursuit + フィルタ用パラメータ
K_LOOK = 1.0#1.5#0.6#1.2           # ルックアヘッド・タイムスケール [s]
V_EPS = 0.5#0.1            # 最低速度下限 [m/s]
MAX_STEER_RAD = 3.1415926535 * 80.0 / 180    # ステア最大角度

# ビュアーやSleepをスキップする高速モード
is_mode_fast = False#True#False




class GenesisScene:
    
    def __init__(self):

#急にエラーで動かなくなった、、、
#   Backend tkagg is interactive backend. Turning interactive mode on.
#   [Genesis] [01:24:48] [WARNING] No Intel XPU device available. Falling back to CPU for torch device.
#   Assertion failed: pCreateInfo->vulkanApiVersion == 0 || (((uint32_t)(pCreateInfo->vulkanApiVersion) >> 22U) == 1 && (((uint32_t)(pCreateInfo->vulkanApiVersion) >> 12U) & 0x3FFU) <= 3), file C:\Users\buildbot\actions-runner\_work\taichi\taichi\external\VulkanMemoryAllocator\include\vk_mem_alloc.h, line 16039
        gs.init(backend=gs.cpu,logging_level="warning")
#gs.init(backend=gs.gpu,logging_level="warning")  # ← CPU でも OK

        # プロンプト生成で町を作る
        if PROMPT_MODE == True:

            # プロンプト例（他にも後で調整可）
            prompt = "a small town with curved roads and a park in the center"

#まだ公開されていない機能
#            # プロンプトからシーン生成
#            self.Scene = generate_scene_from_prompt(prompt)
#
#            # Viewerで可視化
#            viewer = Viewer(self.scene)
#            viewer.run()

        else:

            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(gravity=(0, 0, -9.81)),
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(1, 1, 5),
                    camera_lookat=(0, 0, 0),
                ),
                show_viewer=False if is_mode_fast else True,
            )

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

        if config.REPLAY_MODE:
            self.start_waypoint_idx,self.waypoint_direc = self.get_replay_info()
        else:
            self.start_waypoint_idx = random.randint(0, len(self.waypoints)-1)
            self.waypoint_direc = -1 if random.randint(0,100) < 50 else 1#1で正方向、-1で逆方向

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
        self.debug_arrow_plan = None

        self.debug_plan_arrows = []

        self.debug_wm_arrows = []

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

        self.zero_throttle_time = 0.0
        self.zero_speed_time = 0.0

        #ボトルネック認識とVmax魂の注入
        self.vmax_model = VmaxFactorized(g=9.80665, safety=0.85, r_clip=1000.0, device="cpu")

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

    def get_replay_info(self):
        try:
            with open("replay_info.txt", "r") as f:
                return int(f.readline().strip()),int(f.readline().strip())
        except Exception as e:
            print(f"⚠️ リプレイ情報読み込み失敗: {e}")
            return -int("0"),-int("0")

    # --共通のメソッド(vehicle_control_drlにもある)--

    # selfつけるだけ、ダルすぎる、、どうまとめよう

    def generate_bernoulli_waypoints(
            self,
            num_points: int = 600,
            a: float = 8.0,
            center=np.zeros(3)
    ) -> np.ndarray:
        """
        Continuous Bernoulli lemniscate figure-eight.

        Important:
            半分を作ってmirrorするのではなく、
            t=0〜2π を連続的にサンプリングする。

        Formula:
            x = a√2 cos(t) / (1 + sin(t)^2)
            y = a√2 cos(t) sin(t) / (1 + sin(t)^2)
        """
        import numpy as np
        import math

        t = np.linspace(
            0.0,
            2.0 * math.pi,
            num_points,
            endpoint=False,
            dtype=np.float32,
        )

        sin_t = np.sin(t)
        cos_t = np.cos(t)

        denom = 1.0 + sin_t ** 2

        x = a * math.sqrt(2.0) * cos_t / denom
        y = a * math.sqrt(2.0) * cos_t * sin_t / denom

        pts = np.stack(
            [
                x,
                y,
                np.zeros_like(x),
            ],
            axis=1,
        ).astype(np.float32)

        return pts + np.asarray(center, dtype=np.float32)


    def get_wp_position(self,wp_idx: int,waypoints: np.ndarray):

        wp_idx = wp_idx % len(waypoints)

        return waypoints[wp_idx][:2]


#計画と行動のマルチタスクモデル
    def get_wp_preview(self, K: int = 40) -> np.ndarray:
        """
        前方K点のWPプレビューを車体座標で返す。
        出力: (K,5) 各行 = (dx, dy, s, kappa, width)
        - s    : 先頭からの弧長[m]
        - kappa: 近傍三点からの離散曲率近似
        - width: 路幅（未取得なら定数）
        """
        # 依存：self.scene.waypoints (Nx2), self.scene.start_waypoint_idx, self.scene.waypoint_direc
        waypoints = self.waypoints  # shape (N,2)
        N = waypoints.shape[0]
        idx = int(self.waypoint_idx)
        direc = int(self.waypoint_direc)  # +1 or -1

        # 車体姿勢
#取り方があります        
        ego_pos = np.array(self.car.get_dofs_position())[:2]
        quat = self.car.get_links_quat()[0]  # chassisの回転（w, x, y, z）
        siny_cosp = 2 * (quat[0]*quat[3] + quat[1]*quat[2])
        cosy_cosp = 1 - 2 * (quat[2]**2 + quat[3]**2)
        ego_yaw = math.atan2(siny_cosp, cosy_cosp)
#        ego_pos = self.car.position[:2]    # (x,y)
#        ego_yaw = float(self.car.yaw)      # [rad]

        pts_world = []
        cur = idx
        for i in range(K+2):  # 曲率用に+2
            pts_world.append(self.get_wp_position(cur, waypoints))
            cur = (cur + direc) % N
        pts_world = np.asarray(pts_world)  # (K+2,2)

        # 弧長 s と曲率 kappa を近似
        seg = np.diff(pts_world[:K+1], axis=0)         # (K+1,2)
        ds = np.sqrt((seg**2).sum(axis=1))             # (K+1,)
        s_acc = np.concatenate([[0.0], np.cumsum(ds)]) # (K+2,) 末尾余るがKで切る

        # 曲率: 三点円近似
        kappa = np.zeros(K)
        pwm1 = pts_world[:-2]
        pw   = pts_world[1:-1]
        pwp1 = pts_world[2:]
        v1 = pw - pwm1
        v2 = pwp1 - pw
        cross = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
        d1 = np.sqrt((v1**2).sum(axis=1))
        d2 = np.sqrt((v2**2).sum(axis=1))
        denom = d1 * d2 * np.sqrt(((pwp1 - pwm1)**2).sum(axis=1)) + 1e-8
        kappa[1:-1] = 2.0 * cross[1:-1] / denom[1:-1]
        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]

        # 車体座標へ
        wp_feats = np.zeros((K, 5), dtype=np.float32)
        width = getattr(self, "road_width", 4.0)
        for i in range(K):
            wx, wy = pts_world[i+1]
            dx, dy = self._world_to_ego(wx, wy, ego_pos, ego_yaw)
            wp_feats[i, 0] = dx
            wp_feats[i, 1] = dy
            wp_feats[i, 2] = float(s_acc[i+1])
            wp_feats[i, 3] = float(kappa[i])
            wp_feats[i, 4] = float(width)
        return wp_feats



    # vは正規化不要とのこと
    #roll pitch yaw の順は ROS では標準
    def vector_to_euler(self,v):
        x, y, z = v
        yaw = np.arctan2(y, x)
        pitch = np.arctan2(-z, np.sqrt(x**2 + y**2))
        roll = 0  # ロールは定義できない（方向ベクトルだけでは不定）
        return [roll, pitch, yaw]
#        return np.degrees([roll, pitch, yaw])

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

    #perp_errorを厳密に最近接セグメントから
    def point_to_segment_distance_2d(self,p, a, b):
        """点pと線分abの最短距離と、最近点q、パラメータt(0..1)を返す"""
        ap = p - a
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            q = a
            return float(np.linalg.norm(p - q)), q, 0.0
        t = float(np.dot(ap, ab) / ab2)
        t = max(0.0, min(1.0, t))
        q = a + t * ab
        return float(np.linalg.norm(p - q)), q, t

    #perp_errorを厳密に最近接セグメントから
    def check_near_lateral_distance_to_centerline(self,pos_xy, waypoints_xy, idx_center, direc, window=30):
        """
        近傍window区間だけ探索して、中心線（折れ線）からの最短距離を返す。
        idx_center: 現在追っている waypoint_idx（近傍探索の中心）
        direc: +1 or -1
        """
        wp = np.asarray(waypoints_xy, dtype=np.float32)
        N = wp.shape[0]
        p = np.asarray(pos_xy, dtype=np.float32)

        best_d = 1e9
        best_idx = idx_center

        # idx_center 近傍の線分を探索（全探索より軽い）
        for k in range(-window, window):
            i0 = (idx_center + k * direc) % N
            i1 = (i0 + direc) % N
            a = wp[i0][:2]
            b = wp[i1][:2]
            d, q, _ = self.point_to_segment_distance_2d(p, a, b)
            if d < best_d:
                best_idx = i1
                best_d = d

        return best_idx



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
        if config.REPLAY_MODE is None:
            self.waypoint_direc = -1 if random.randint(0,100) < 50 else 1#1で正方向、-1で逆方向
            self.start_waypoint_idx = random.randint(0, len(self.waypoints)-1)

        self.end_waypoint_idx = (self.start_waypoint_idx - self.waypoint_direc) % len(self.waypoints)
        self.waypoint_idx = self.start_waypoint_idx

        # 車の初期位置
        self.set_car_start_pos(self.car,self.start_waypoint_idx,self.waypoint_direc)

        # 報酬リセット
        self.reward_total = 0.0

        self.zero_throttle_time = 0.0
        self.zero_speed_time = 0.0

#        # 最初からワイヤーにしたい
#        if self.scene.viewer is not None:
#            time.sleep(1)  # Viewerが起動するまで待つ
#            pyautogui.press('d')  # ワイヤーフレームトグル

#        #ボトルネック認識とVmax魂の注入
#        self.vmax_model = VmaxFactorized(g=9.80665, safety=0.85, r_clip=1000.0, device="cpu")

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

#        print(f"推論時の control_dofs_force throttle={throttle}")

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
        elif self.t > 10.0 and self.reward_total <= 0:
            # 成功の可能性が低い
            reward -= 1000
            done = True            
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
        elif self.zero_throttle_time > 1.0:
            # ずっとアクセルを踏んでいない
            reward -= 100
            # ペナルティは払ったのでクリアする
            self.zero_throttle_time = 0.0
#            done = True
        elif self.zero_speed_time > 1.0:
            # ずっとアクセルを踏んでいない
            reward -= 100
            # ペナルティは払ったのでクリアする
            self.zero_speed_time = 0.0
#            done = True

        # アクセルを踏んでいない時間
        if throttle <=0.05:
            self.zero_throttle_time += self.dt
        else:
            self.zero_throttle_time = 0.0
            
        # 速度が出ていない時間
        if obs[6] <=0.5:
            self.zero_speed_time += self.dt
        else:
            self.zero_speed_time = 0.0

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

    def ego_to_world_batch(self,plan_xy: np.ndarray, pos_xy: np.ndarray, yaw: float) -> np.ndarray:
        """
        plan_xy: (2M,) = [x1,y1,...]  (ego座標, x前方/ y左+)
        pos_xy : (2,)   vehicle world position (x,y)
        yaw    : float  vehicle yaw [rad]
        return : (M,2) world XY
        """
        assert plan_xy.ndim == 1 and plan_xy.size % 2 == 0
        M = plan_xy.size // 2
        pts = plan_xy.reshape(M, 2).astype(np.float32)  # (M,2) [ex,ey]
        c, s = np.cos(yaw), np.sin(yaw)
        # world = R(yaw) @ ego + pos
        wx = c * pts[:, 0] - s * pts[:, 1] + pos_xy[0]
        wy = s * pts[:, 0] + c * pts[:, 1] + pos_xy[1]
        return np.stack([wx, wy], axis=1)  # (M,2)



    def _get_pose_world_xy_yaw(self):
        """chassisリンクから位置(x,y)とyawを一貫フレームで取得"""
        import math, numpy as np
        pos_w = np.array(self.car.get_links_pos()[0], dtype=np.float32)  # (x,y,z) world
        quat  = self.car.get_links_quat()[0]  # (w,x,y,z)
        w,x,y,z = quat
        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return pos_w[:2], yaw

    def _ego_to_world(self, pts_ego_xy, pos_world_xy, yaw):
        """pts_ego_xy: (N,2), pos_world_xy: (2,), yaw[rad]"""
        import numpy as np, math
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s],
                    [s,  c]], dtype=np.float32)  # x前方, y左を仮定
        return pos_world_xy + pts_ego_xy @ R.T

    def debug_draw_plan_compare(self, plan_pred_1d, plan_gt_1d=None):
        """
        plan_pred_1d: (2M,) 予測計画（ego座標）
        plan_gt_1d:  (2M,) GT計画（ego座標）※任意
        worldに変換して線分を描く
        """
        import numpy as np
        plan_pred = np.asarray(plan_pred_1d, np.float32).reshape(-1)
        assert plan_pred.size % 2 == 0, "plan_pred must be (2M,)"
        P_pred = plan_pred.reshape(-1, 2)  # (M,2)

        P_gt = None
        if plan_gt_1d is not None:
            arr_gt = np.asarray(plan_gt_1d, np.float32).reshape(-1)
            assert arr_gt.size % 2 == 0, "plan_gt must be (2M,)"
            P_gt = arr_gt.reshape(-1, 2)  # (M,2)

        # 一貫フレームで pose を取得
        pos_xy, yaw = self._get_pose_world_xy_yaw()

        # ego→world
        W_pred = self._ego_to_world(P_pred, pos_xy, yaw)  # (M,2)
        W_gt   = self._ego_to_world(P_gt,   pos_xy, yaw) if P_gt is not None else None

        # まず既存の矢印をクリア（必要なら保持して個別に消す）
        try:
            if getattr(self, "_dbg_arrows_pred", None):
                for h in self._dbg_arrows_pred: self.scene.clear_debug_object(h)
            if getattr(self, "_dbg_arrows_gt", None):
                for h in self._dbg_arrows_gt: self.scene.clear_debug_object(h)
        except Exception:
            pass
        self._dbg_arrows_pred, self._dbg_arrows_gt = [], []

        # 予測（シアン）
        for i in range(max(0, W_pred.shape[0]-1)):
            src = W_pred[i]; dst = W_pred[i+1]; vec = dst - src
            h = self.scene.draw_debug_arrow(
                pos=(float(src[0]), float(src[1]), 0.10),
                vec=(float(vec[0]), float(vec[1]), 0.00),
                radius=0.005, color=(0.0, 1.0, 1.0, 0.6) # cyan
            )
            self._dbg_arrows_pred.append(h)

        # GT（青）
        if W_gt is not None:
            for i in range(max(0, W_gt.shape[0]-1)):
                src = W_gt[i]; dst = W_gt[i+1]; vec = dst - src
                h = self.scene.draw_debug_arrow(
                    pos=(float(src[0]), float(src[1]), 0.10),
                    vec=(float(vec[0]), float(vec[1]), 0.00),
                    radius=0.005, color=(0.0, 0.0, 1.0, 0.6) # blue
                )
                self._dbg_arrows_gt.append(h)

        # --- 最小自己診断（座標変換の健全性チェック） ---
        # ego の (1,0) → 車前方へ，(0,1) → 車左へ になっているか？
        unit = np.array([[1,0],[0,1]], np.float32)
        W_unit = self._ego_to_world(unit, pos_xy, yaw)
        # 必要なら print して確認
        # print(f"[DBG] yaw={yaw:.3f}  fwd_world={W_unit[0]-pos_xy}  left_world={W_unit[1]-pos_xy}")

    def debug_draw_plan_xy(self, plan_xy_1d):
        """
        plan_xy_1d: shape (2M,) の 1D 配列（ego座標: [x1,y1,x2,y2,...]）
        """
        import math
        import numpy as np

        arr = np.asarray(plan_xy_1d, dtype=np.float32).reshape(-1)
        assert arr.size % 2 == 0, "plan shape must be (2M,)"

        M = arr.size // 2
        if M < 2:
            return  # 線分が作れない

        pts_ego = arr.reshape(M, 2)  # (M,2)

        # 車の現在姿勢（world）
        pos_xy = np.array(self.car.get_dofs_position()[:2], np.float32)  # (2,)
        quat   = self.car.get_links_quat()[0]  # (w,x,y,z)
        siny_cosp = 2*(quat[0]*quat[3] + quat[1]*quat[2])
        cosy_cosp = 1 - 2*(quat[2]**2 + quat[3]**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # ego → world 変換
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s],
                    [s,  c]], dtype=np.float32)
        pts_world = pos_xy + pts_ego @ R.T  # (M,2)

        # 複数本の矢印を 1 本ずつ描く（vec は (3,) のフラットな float タプル）
        for debug_plan_arrow in self.debug_plan_arrows:
            self.scene.clear_debug_object(debug_plan_arrow)

        self.debug_plan_arrows.clear()

        for i in range(M - 1):
            src = pts_world[i]
            dst = pts_world[i + 1]
            v = dst - src  # (2,)

            debug_obj = self.scene.draw_debug_arrow(
                pos=(float(src[0]), float(src[1]), 0.10),
                vec=(float(v[0]),   float(v[1]),   0.00),
                radius=0.005,
                color=(0.0, 1.0, 1.0, 0.5),
            )
            self.debug_plan_arrows.append(debug_obj)


    #損失に世界モデルを使う　可視化
    def debug_draw_world_modeling(self, wm_debug_dict, scale=2.0):
        """
        World Modelの1step予測をデバッグ描画する。

        wm_debug_dict:
            wm_runtime.debug_dict(...) の戻り値

        表示内容:
            - 現在位置から、予測perp変化方向へ小さい矢印
            - heading予測方向の矢印
        """
        import math
        import numpy as np

        if not hasattr(self, "debug_wm_arrows"):
            self.debug_wm_arrows = []

        # 既存WM矢印をクリア
        for obj in self.debug_wm_arrows:
            self.scene.clear_debug_object(obj)
        self.debug_wm_arrows.clear()

        perp_now = float(wm_debug_dict["perp_now"])
        perp_next = float(wm_debug_dict["wm_perp_next"])
        head_now = float(wm_debug_dict["heading_now"])
        head_next = float(wm_debug_dict["wm_heading_next"])

        d_perp = perp_next - perp_now
        d_head = head_next - head_now

        # 車の現在姿勢 world
        pos_xy = np.array(self.car.get_dofs_position()[:2], np.float32)

        quat = self.car.get_links_quat()[0]  # (w,x,y,z)
        siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
        cosy_cosp = 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        c, s = math.cos(yaw), math.sin(yaw)

        # 車体前方向・左方向
        forward = np.array([c, s], dtype=np.float32)
        left = np.array([-s, c], dtype=np.float32)

        # --------------------------------------------------
        # 1. perp_error変化の矢印
        # --------------------------------------------------
        # perp_errorが増える方向を左方向として描く
        # 符号が逆に見えたら -left にしてください
        # 推論が正しければ常に道路の中央を指す
#はずれをストレートに表示
        perp_vec_xy = left * (perp_next * scale)
#        perp_vec_xy = -left * (d_perp * scale)

        debug_obj = self.scene.draw_debug_arrow(
            pos=(float(pos_xy[0]), float(pos_xy[1]), 0.30),
            vec=(float(perp_vec_xy[0]), float(perp_vec_xy[1]), 0.00),
            radius=0.01,
            color=(1.0, 0.0, 1.0, 0.8),  # magenta
        )
        self.debug_wm_arrows.append(debug_obj)

        # --------------------------------------------------
        # 2. heading予測方向の矢印
        # --------------------------------------------------
        # head_nextを車体yawに足して、予測姿勢方向として表示
        pred_yaw = yaw + d_head

        cp, sp = math.cos(pred_yaw), math.sin(pred_yaw)
        pred_forward = np.array([cp, sp], dtype=np.float32)

        heading_arrow_len = 1.0

        start = pos_xy + forward * 0.5

        debug_obj = self.scene.draw_debug_arrow(
            pos=(float(start[0]), float(start[1]), 0.45),
            vec=(
                float(pred_forward[0] * heading_arrow_len),
                float(pred_forward[1] * heading_arrow_len),
                0.00,
            ),
            radius=0.01,
            color=(1.0, 1.0, 0.0, 0.8),  # yellow
        )
        self.debug_wm_arrows.append(debug_obj)

        # --------------------------------------------------
        # 3. 危険度補助: outside方向が強い場合は赤矢印
        # --------------------------------------------------
        if abs(perp_next) > 1.0:
            risk_vec_xy = left * (np.sign(perp_next) * 0.8)

            debug_obj = self.scene.draw_debug_arrow(
                pos=(float(pos_xy[0]), float(pos_xy[1]), 0.60),
                vec=(float(risk_vec_xy[0]), float(risk_vec_xy[1]), 0.00),
                radius=0.015,
                color=(1.0, 0.0, 0.0, 0.9),  # red
            )
            self.debug_wm_arrows.append(debug_obj)


    def compute_perp_error(self,pos, wp0, wp1, lane_half_width=2.0):
        segment = wp1 - wp0
        seg_len = np.linalg.norm(segment)

        if seg_len < 1e-6:
            return 0.0

        seg_dir = segment / seg_len
        rel = pos - wp0

        # 符号付き横ずれ[m]
        signed_dist = rel[0] * seg_dir[1] - rel[1] * seg_dir[0]

        # 車線半幅で正規化
        perp_error = signed_dist / lane_half_width

        return float(perp_error)

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

        # セグメント距離
        segment_len = np.linalg.norm(segment)
        if segment_len > 0.001:
            scale = 0.5 * (1.0 / segment_len)
        else:
            scale = 1.0

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


#前に進まないので調査

        # ヘディング誤差（-pi ～ +pi に wrap）
        # ※車体とターゲット方向の角度差
        heading_error = target_yaw - yaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # perp_errorは道路方向とのずれ

#perp_errorを厳密に最近接セグメントから
        near_waypoint_idx = self.check_near_lateral_distance_to_centerline(
            pos_xy=pos,
            waypoints_xy=self.waypoints,   # (N,2)
            idx_center=self.waypoint_idx,
            direc=self.waypoint_direc * (-1), #手前側に探索
            window=40
        )
        near_wp = self.get_wp_position(near_waypoint_idx,self.waypoints)
        near_next_wp = self.get_wp_position(near_waypoint_idx+self.waypoint_direc,self.waypoints)
        perp_error = self.compute_perp_error(pos, near_wp, near_next_wp, lane_half_width=2.0)
#        target_dir = target_wp - pos
#        target_dir = target_dir / np.linalg.norm(target_dir)
#        segment_dir = segment / np.linalg.norm(segment)
#        inner_angle = np.dot(target_dir, segment_dir)
#        perp_error = 1.0 - inner_angle

        #まだチェックポイント上に乗っていないのでコースアウトは無視する
        if self.waypoint_idx == self.start_waypoint_idx:
            perp_error = 0.0

		#計画と行動のマルチタスクモデル 計画が相対なのでターゲット位置も相対に変更
        target_wp_subvec = target_wp - pos
        target_wp_relative_x =  math.cos(yaw)*target_wp_subvec[0] + math.sin(yaw)*target_wp_subvec[1]
        target_wp_relative_y = -math.sin(yaw)*target_wp_subvec[0] + math.cos(yaw)*target_wp_subvec[1]

        car_yaw_sin,car_yaw_cos = yaw_to_sin_cos(yaw)

#ボトルネック認識とVmax魂の注入

        # --- _get_obs() 内：VMAX 系観測を作る安全な順序 ---

        # 1) 先読み長を決定
        H = int(getattr(self, "H_preview", 20))
        assert H >= 1, f"H_preview must be >=1 (got {H})"

        # 2) 先読みの曲率列と区間長を計算（スカラーを返す実装にしておく）
#共通化
        kappas_H, ds_H = curvature_from_wps(self.waypoints, self.waypoint_idx, self.waypoint_direc, H)
#        kappas_H, ds_H = self._curvature_from_wps(self.waypoint_idx, self.waypoint_direc, H)

        # 形の揺れ対策：必ず 1D float 配列へ
        kappas_H = np.asarray(kappas_H, dtype=np.float32).reshape(-1)
        assert kappas_H.size >= 1, "curvature list must be non-empty"

        # 3) μ（路面摩擦）列を用意（当面は固定。将来は路面タグから取得）
        mu_default = getattr(self, "mu_default", 0.8)
        mus_H = np.full_like(kappas_H, fill_value=float(mu_default), dtype=np.float32)

        # 4) 局所値を決定
        kappa_local = float(kappas_H[0])
        mu_local    = float(mus_H[0])

        # 5) 学習済み v_max モデルで推論（単点 & 先読み列）
        vmax_local   = float(self.vmax_model.from_kappa(kappa_local, mu_local))       # 単点
        vmax_preview = np.asarray(self.vmax_model.batch_kappa(kappas_H, mus_H), np.float32)  # (H,)

        # 6) 先読みから要約量
        if vmax_preview.size >= 2:
            vmax_min_hH   = float(np.min(vmax_preview))
            vmax_mean_hH  = float(np.mean(vmax_preview))
            vmax_slope_hH = float((vmax_preview[-1] - vmax_preview[0]) / (vmax_preview.size - 1))
        else:
            vmax_min_hH = vmax_mean_hH = float(vmax_local)
            vmax_slope_hH = 0.0

        # 7) 制限速度と統合（現状は∞→ v_max_min がそのままターゲット）
        speed_limit    = float("inf")   # TODO: マップ側から取得
        limit_v_target = float(min(speed_limit, vmax_min_hH))

#        print(f"[VMAX] vel={vel:.3f}  vmax_local={vmax_local:.3f}  "
#            f"vmax_min_hH={vmax_min_hH:.3f}  limit_v_target={limit_v_target:.3f}  "
#            f"v_ratio={float(vel)/(vmax_local+1e-3):.3f}  headroom={vmax_local-float(vel):.3f}")

        # 8) 既存10次元（obs10）を先に構築してある前提
        #   obs10 = np.array([target_wp_relative_x, target_wp_relative_y, pos[0], pos[1],
        #                     car_yaw_sin, car_yaw_cos, vel, perp_error, heading_error, passed], np.float32)
        # --- 既存の10次元をまず作る（あなたの既存コード） ---
        obs10 = np.array([
            target_wp_relative_x, target_wp_relative_y, pos[0], pos[1],
            car_yaw_sin,car_yaw_cos, vel, perp_error, heading_error, passed
        ], dtype=np.float32)

        # 9) 追加9次元（VMAX塊）
        obs_extra = np.array([
            kappa_local, mu_local, vmax_local,
            float(vel) / (vmax_local + 1e-3),          # v_ratio
            vmax_local - float(vel),                   # headroom
            vmax_min_hH, vmax_mean_hH, vmax_slope_hH,
            limit_v_target
        ], dtype=np.float32)

        # 10) （任意）デバッグ／学習用のキャッシュ
        self._cache_kappas_H = kappas_H.astype(np.float32, copy=False)
        self._cache_ds_H     = np.asarray(ds_H, np.float32)
        self._cache_vmax_H   = vmax_preview

        # 11) 連結して OBS_V2(19次元) を返す
        obs_v2 = np.concatenate([obs10, obs_extra], axis=0)   # shape=(19,)
        return obs_v2
#
##計画と行動のマルチタスクモデル 計画が相対なのでターゲット位置も相対に変更
#        return np.array([target_wp_relative_x, target_wp_relative_y, pos[0], pos[1], car_yaw_sin,car_yaw_cos, vel, perp_error, heading_error,passed], dtype=np.float32)
##        return np.array([target_wp[0], target_wp[1], pos[0], pos[1], car_yaw_sin,car_yaw_cos, vel, perp_error, heading_error,passed], dtype=np.float32)


#計画と行動のマルチタスクモデル
    def _world_to_ego(self, px: float, py: float, ego_pos: np.ndarray, ego_yaw: float) -> Tuple[float, float]:
        dx = px - ego_pos[0]
        dy = py - ego_pos[1]
        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        ex = c*dx - s*dy
        ey = s*dx + c*dy
        return ex, ey




    def _compute_reward(self, obs, t):
        # obs = [x, y, yaw, speed, cross_track_err, heading_err]
        speed = obs[5+1]
        cte   = obs[6+1]  # Cross Track Error
        he    = obs[7+1]  # Heading Error
        passed = obs[8+1] #　ポイント通過

        # 基本報酬：速度を奨励しつつ、軌道逸脱を罰する

        speed = speed * math.cos(he)
#ボトルネック認識とVmax魂の注入 早い程良いに決まってるので戻す
#        # いやだけどこうしないとスピード狂がどうにもやめられない
#        speed = 20.0 if speed > 20.0 else speed 

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

        # さすがに出しすぎ
        if speed > 300.0:
            reward -= 10.0
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

#計画と行動のマルチタスクモデル
        self.scene = GenesisScene()
        # 実観測からshapeを自動決定
        obs0 = self.scene._get_obs().astype(np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32)
        # アクション範囲（自前フィールドも保持）
        self._act_low  = np.array([-1.0,  0.0], dtype=np.float32)   # steer∈[-1,1], throttle∈[0,1]
        self._act_high = np.array([ 1.0,  1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=self._act_low, high=self._act_high, dtype=np.float32)
#        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),
#                                       high=np.array([1.0, 1.0]), dtype=np.float32)
#
#        # _get_obs の要素を増やしたら、shapeの数を増やす必要がある
#        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
#
#        self.scene = GenesisScene()


#計画と行動のマルチタスクモデル
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Gymnasium API: (obs, info) を返す"""
        super().reset(seed=seed)
        scene_seed = getattr(self.scene, "seed", None)
        if seed is not None and callable(scene_seed):
            scene_seed(seed)
        self.scene.reset()
        obs = self.scene._get_obs().astype(np.float32)
        return obs, {}
#    def reset(self):
#        return self.scene.reset()


#計画と行動のマルチタスクモデル
    def step(self, action):
        """Gymnasium API: (obs, reward, terminated, truncated, info)"""
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, self._act_low, self._act_high)
        steer, throttle = float(a[0]), float(a[1])
        # 既存シーンが (obs, reward, done, info) を返す想定 → Gymnasium形式に変換
        obs, reward, done, info = self.scene.step(steer, throttle)
        terminated = bool(done)
        truncated  = bool(info.get("truncated", False))
        return np.asarray(obs, dtype=np.float32), float(reward), terminated, truncated, info
#    def step(self, action):
#        steer, throttle = action
#        return self.scene.step(steer, throttle)

    def close(self):
        self.scene.close()

