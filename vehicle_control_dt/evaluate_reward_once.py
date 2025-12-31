import os

import torch

import pickle
import argparse
import numpy as np

from model_dt import DecisionTransformer
from train_dt import SequenceDataset  # または TrajectoryDataset
from genesis_gym_env import GenesisEnv  # 必要に応じて調整

#ボトルネック認識とVmax魂の注入 7.1 外部ノーマライザは使わない
#from utils.trajectory_utils import normalize

#計画と行動のマルチタスクモデル
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Tuple

TIMESTEP_MAX = 4000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#計画と行動のマルチタスクモデル
K_WP = 40
PLAN_M = 3


# ここで暫定モデル temp_model.pt がロードされる
CHECKPOINT_PATH = "checkpoints/temp_model.pt"
PKL_PATH = "data_dt/trajectories_dt.pkl"
NORM_PATH = "data_dt/mean_std.pkl"

INITIAL_RTG = 100.0
USE_FIXED_RTG = True









#計画と行動のマルチタスクモデル
class Normalizer:
    def __init__(self, path):
        with open(path, "rb") as f:
            stats = pickle.load(f)
        # 必須キー
        self.obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)
        self.obs_std  = np.asarray(stats["obs_std"],  dtype=np.float32)
        # RTGは ret_* / rtg_* どちらでも拾えるように
        self.rtg_mean = np.float32(stats.get("ret_mean", stats.get("rtg_mean", 0.0)))
        self.rtg_std  = np.float32(stats.get("ret_std",  stats.get("rtg_std",  1.0)))
        # 追記：WPの正規化情報（無ければ単位スケール）
        self.wp_mean  = np.asarray(stats.get("wp_mean", np.zeros(5, dtype=np.float32)), dtype=np.float32)
        self.wp_std   = np.asarray(stats.get("wp_std",  np.ones(5,  dtype=np.float32)), dtype=np.float32)

    @staticmethod
    def normalize(x, mean, std, eps: float = 1e-6):
        return (x - mean) / (std + eps)

    def normalize_obs(self, obs):
        return self.normalize(obs, self.obs_mean, self.obs_std)

    def normalize_rtg(self, rtg):
        return self.normalize(rtg, self.rtg_mean, self.rtg_std)

    def normalize_wp(self, wp):
        return self.normalize(wp, self.wp_mean, self.wp_std)
#
#class Normalizer:
#    def __init__(self, path):
#
#        with open(path, "rb") as f:
#            stats = pickle.load(f)
#        self.obs_mean, self.obs_std = stats["obs_mean"], stats["obs_std"]
#        self.rtg_mean, self.rtg_std = stats["ret_mean"], stats["ret_std"]
#
##計画と行動のマルチタスクモデル 追記：WPの正規化情報（なければ単位スケール）
#        self.wp_mean = d.get("wp_mean", np.zeros(5, dtype=np.float32))
#        self.wp_std  = d.get("wp_std",  np.ones(5, dtype=np.float32))
#
##計画と行動のマルチタスクモデル
#    def normalize(x, mean, std, eps=1e-6):
#        return (x - mean) / (std + eps)
#
#    def normalize_obs(self, obs):
#        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
#
#    def normalize_rtg(self, rtg):
#        return (rtg - self.rtg_mean) / (self.rtg_std + 1e-8)


#最新モデルでリプレイする　別手法
def run_inference_once(context_len, n_layer, n_head,norm_path,pkl_path,checkpoint_path):
#def run_inference_once(context_len, n_layer, n_head):

 #最新モデルでリプレイする　別手法
    norm = Normalizer(norm_path)
 #   norm = Normalizer(NORM_PATH)

    # === DTデータ読み込み（1件目から形式確認用）

 #最新モデルでリプレイする　別手法
    with open(pkl_path, "rb") as f:
#    with open(PKL_PATH, "rb") as f:
        trajs = pickle.load(f)
    traj = trajs[0]

    print("=== DEBUG ===")
    print("type(traj):", type(traj))
    print("keys:", traj.keys())
    print("type of traj['observations']:", type(traj["observations"]))
    print("shape of traj['observations']:", np.shape(traj["observations"]))
    print("obs_dim:", traj["observations"].shape[1])
    print("act_dim:", traj["actions"].shape[1])
    print("context_len:", context_len)
    print("obs_mean, obs_std:", norm.obs_mean, norm.obs_std)
    print("rtg_mean, rtg_std:", norm.rtg_mean, norm.rtg_std)

    # モデル生成
    model = DecisionTransformer(
        obs_dim=traj["observations"].shape[1],
        act_dim=traj["actions"].shape[1],
#計画と行動のマルチタスクモデル
        context_len=context_len,
        n_layer=n_layer,
        n_head=n_head,
        plan_M=PLAN_M,
        use_focus=False
#       context_len=context_len,
#        n_layer=n_layer,
#        n_head=n_head
    ).to(DEVICE)


#ボトルネック認識とVmax魂の注入 7.2 
    assert traj["observations"].shape[1] == 19, f"OBS_V2(19) 想定。got {traj['observations'].shape[1]}"
    # map_location を明示（CPUでもOKに）
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
#    model.load_state_dict(torch.load(checkpoint_path))

    model.eval()

	#ボトルネック認識とVmax魂の注入 アクセルが負の数値になる
    # === Probe 1: zero-input 出力を確認（埋め込み＋出力層の“素の向き”をチェック）===
    with torch.no_grad():
        B, T = 1, 1
        t = torch.zeros(B, T, dtype=torch.long, device=DEVICE)
        s = torch.zeros(B, T, model.obs_dim, device=DEVICE)
        a = torch.zeros(B, T, model.act_dim, device=DEVICE)
        r = torch.zeros(B, T, 1,             device=DEVICE)
        # WPは何も与えない（分布影響を切り分けるため）
        pa0, _, _ = model(t, s, a, r, wp=None, return_plan=False)
        print("[CHK] zero-input pred =", pa0[0, 0].detach().cpu().numpy())
    

    #ボトルネック認識とVmax魂の注入 7.3 
    torch.set_grad_enabled(False)

    # === 環境初期化 ===

#計画と行動のマルチタスクモデル
    # === 環境初期化（Gymnasium専用）===
    env = GenesisEnv()
    obs, _ = env.reset(seed=0)
#    env = GenesisEnv()
#    obs = env.reset()

    done = False
    total_reward = 0.0
    t = 0

#ボトルネック認識とVmax魂の注入 7.4 過去文脈をゼロ埋めするより「直近obsの複製」の方が安定
    obs_buffer = [obs.astype(np.float32)] * context_len
#    obs_buffer = [obs] * context_len


    act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
    timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

    # DTのMLP化検証 復元step8

# pklから初期報酬を渡す    
    initial_rtg = np.array(traj["initial_rtg"][0], dtype=np.float32)  # shape: (1,)
#    initial_rtg = np.array([INITIAL_RTG], dtype=np.float32)  # shape: (1,)
    rtg_buffer = [initial_rtg.copy() for _ in range(context_len)]

    print(f"✅ 目標報酬を設定: {initial_rtg[0]}")

#計画と行動のマルチタスクモデル 1回だけWP正規化のmean/stdをtorch化
    wp_mean_t = torch.tensor(norm.wp_mean, dtype=torch.float32, device=DEVICE)
    wp_std_t  = torch.tensor(norm.wp_std,  dtype=torch.float32, device=DEVICE)

    #ボトルネック認識とVmax魂の注入 7.5 ログ用: 制約違反率（vel>limit_v_target）
    viol_cnt = 0
    steps_cnt = 0

    for t in range(100_000):

        # 正規化＋テンソル化

#ボトルネック認識とVmax魂の注入 7.6 正規化（学習と同一の mean/std を使う）
        obs_norm = norm.normalize_obs(np.array(obs_buffer, dtype=np.float32))   # (T,19)
        rtg_norm = norm.normalize_rtg(np.array(rtg_buffer, dtype=np.float32))   # (T,1)
#        obs_norm = normalize(np.array(obs_buffer), norm.obs_mean, norm.obs_std)
#        rtg_norm = normalize(np.array(rtg_buffer), norm.rtg_mean, norm.rtg_std)

        actions_np = np.array(act_buffer)
        ts = np.array(timestep_buffer)

        obs_tensor = torch.tensor(obs_norm.copy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        act_tensor = torch.tensor(actions_np.copy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        rtg_tensor = torch.tensor(rtg_norm.copy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        ts_tensor  = torch.tensor(ts.copy(), dtype=torch.long).unsqueeze(0).to(DEVICE)

#ボトルネック認識とVmax魂の注入 7.7 WPプレビュー（無い環境でも落ちないようにガード）
        wp_np = getattr(env.scene, "get_wp_preview", lambda k: None)(K_WP)
        if wp_np is not None and len(wp_np) > 0:
            wp_tensor = torch.tensor(wp_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,K,5)
            wp_tensor = (wp_tensor - wp_mean_t) / (wp_std_t + 1e-6)
        else:
            wp_tensor = None
#        wp_np = env.scene.get_wp_preview(K_WP)                         # (K,5)
#        wp_tensor = torch.tensor(wp_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,K,5)
#        wp_tensor = (wp_tensor - wp_mean_t) / (wp_std_t + 1e-6)


#計画と行動のマルチタスクモデル
        with torch.no_grad():
            action_pred, plan_hat, alpha = model(
                ts_tensor, obs_tensor, act_tensor, rtg_tensor,
                wp=wp_tensor, return_plan=True, return_focus=False
            )
            action = action_pred[0, -1].cpu().numpy()

#ボトルネック認識とVmax魂の注入 7.8 行動クリップ（環境の許容範囲に合わせて調整）
            action[0] = float(np.clip(action[0], -1.0, 1.0))  # steer
            action[1] = float(np.clip(action[1], -1.0, 1.0))  # accel/brake
            # 計画の可視化
            debug_plan_xy = [0.0,0.0]
            if plan_hat is not None:
                # (1,2M) or (1,T,2M) の両方に対応
                if plan_hat.dim() == 3:
                    debug_plan_xy = plan_hat[0, -1].cpu().numpy()   # (2M,)
                else:
                    debug_plan_xy = plan_hat[0].cpu().numpy()       # (2M,)
                env.scene.debug_draw_plan_xy(debug_plan_xy)

        # step: Gymnasium専用（5タプル）
        obs, reward, terminated, truncated, info = env.step(action)


        done = bool(terminated) | bool(truncated)
#            
#        with torch.no_grad():
#           action_pred = model(ts_tensor, obs_tensor, act_tensor, rtg_tensor)
#           action = action_pred[0, -1].cpu().numpy()
#        # 実行
#        obs, reward, done, _ = env.step(action)

        total_reward += reward

        # バッファ更新
        obs_buffer.pop(0)
#ボトルネック認識とVmax魂の注入 7.9 
        obs_buffer.append(obs.astype(np.float32))
#        obs_buffer.append(obs)

        act_buffer.pop(0)
        act_buffer.append(action)

        if not USE_FIXED_RTG:
            #最低２つないとpopで空になる
            if len(rtg_buffer) >= 2:
                rtg_buffer.pop(0)
                rtg_buffer.append(rtg_buffer[-1] - reward)  # 累積リターン更新

        timestep_buffer.pop(0)
        timestep_buffer.append(t % TIMESTEP_MAX)  # timestepは最大1024まで（Embedding制約）

        #ボトルネック認識とVmax魂の注入 7.9 ログ: 制約違反カウント（OBS_V2: vel=idx6, limit=idx18）
        steps_cnt += 1
        try:
            if obs[6] > obs[18]:
                viol_cnt += 1
        except Exception:
            pass

        if done:
            print(f"✅ 終了ステップ数: {t}")
            print(f"✅ リプレイ情報の記録: start_waypoint_idx={env.scene.start_waypoint_idx} waypoint_direc={env.scene.waypoint_direc}")
            with open("replay_info.txt", "w") as f:
                 f.write(str(env.scene.start_waypoint_idx)+'\n')
                 f.write(str(env.scene.waypoint_direc)+'\n')
                 
            #ボトルネック認識とVmax魂の注入 7.10 ログ: 制約違反カウント（OBS_V2: vel=idx6, limit=idx18）
            viol_rate = (viol_cnt / max(1, steps_cnt))
            print(f"✅ 速度制約違反率: {viol_rate:.3f}")
            with open("eval_score.txt", "w") as f:
                f.write(f"total_reward={total_reward:.4f}\n")
                f.write(f"violation_rate={viol_rate:.6f}\n")

            break

    return total_reward


if __name__ == "__main__":

    ignore_arg = False

    if ignore_arg:

        # 即時確認用
#最新モデルでリプレイする　別手法
        score = run_inference_once(1,2,4,NORM_PATH,PKL_PATH,CHECKPOINT_PATH)
#        score = run_inference_once(1,2,4)


        print(f"評価スコア: {score:.2f}")
        with open("eval_score.txt", "w") as f:
            f.write(str(score))

    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--context_len", type=int, required=True)
        parser.add_argument("--n_layer", type=int, required=True)
        parser.add_argument("--n_head", type=int, required=True)
        #最新モデルでリプレイする　別手法
        parser.add_argument("--norm_path", type=str, required=True)
        parser.add_argument("--pkl_path", type=str, required=True)
        parser.add_argument("--checkpoint_path", type=str, required=True)

        args = parser.parse_args()

        ignore_error = True

        if ignore_error:
            # 例外無視用
            score = run_inference_once(args.context_len, args.n_layer, args.n_head,
                                        args.norm_path,args.pkl_path,args.checkpoint_path)
            print(f"評価スコア: {score:.2f}")
            with open("eval_score.txt", "w") as f:
                f.write(str(score))
        else:
            # 例外厳密に処理
            try:
                score = run_inference_once(args.context_len, args.n_layer, args.n_head,
                                           args.norm_path,args.pkl_path,args.checkpoint_path)
                print(f"評価スコア: {score:.2f}")
                with open("eval_score.txt", "w") as f:
                    f.write(str(score))
            except Exception as e:
                print(f"⚠️ 評価に失敗: {e}")
                with open("eval_score.txt", "w") as f:
                    f.write("")
