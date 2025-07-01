import os

import torch

import pickle
import argparse
import numpy as np

from model_dt import DecisionTransformer
from train_dt import SequenceDataset  # または TrajectoryDataset
from genesis_gym_env import GenesisEnv  # 必要に応じて調整
from utils.trajectory_utils import normalize


TIMESTEP_MAX = 4000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ここで暫定モデル temp_model.pt がロードされる
CHECKPOINT_PATH = "checkpoints/temp_model.pt"
PKL_PATH = "data_dt/trajectories_dt.pkl"
NORM_PATH = "data_dt/mean_std.pkl"

INITIAL_RTG = 100.0
USE_FIXED_RTG = True


class Normalizer:
    def __init__(self, path):

# inference_dtに合わせないと型例外でる
        with open(path, "rb") as f:
            stats = pickle.load(f)
        self.obs_mean, self.obs_std = stats["obs_mean"], stats["obs_std"]
        self.rtg_mean, self.rtg_std = stats["ret_mean"], stats["ret_std"]
#        with open(path, "rb") as f:
#            data = pickle.load(f)
#        self.obs_mean = torch.tensor(data["obs_mean"], dtype=torch.float32)
#        self.obs_std = torch.tensor(data["obs_std"], dtype=torch.float32)
#        self.rtg_mean = torch.tensor(data.get("ret_mean", 0.0), dtype=torch.float32)
#        self.rtg_std = torch.tensor(data.get("ret_std", 1.0), dtype=torch.float32)

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def normalize_rtg(self, rtg):
        return (rtg - self.rtg_mean) / (self.rtg_std + 1e-8)


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
        context_len=context_len,
        n_layer=n_layer,
        n_head=n_head
    ).to(DEVICE)

 #最新モデルでリプレイする　別手法
    model.load_state_dict(torch.load(checkpoint_path))
#    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
    model.eval()

    # === 環境初期化 ===
    env = GenesisEnv()
    obs = env.reset()


    done = False
    total_reward = 0.0
    t = 0

    # DTのMLP化検証 復元step8
    obs_buffer = [obs] * context_len
    act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
    timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

    # DTのMLP化検証 復元step8

# pklから初期報酬を渡す    
    initial_rtg = np.array(traj["initial_rtg"][0], dtype=np.float32)  # shape: (1,)
#    initial_rtg = np.array([INITIAL_RTG], dtype=np.float32)  # shape: (1,)
    rtg_buffer = [initial_rtg.copy() for _ in range(context_len)]

    print(f"✅ 目標報酬を設定: {initial_rtg[0]}")

    for t in range(100_000):

        # 正規化＋テンソル化
# inference_dtに合わせる
        obs_norm = normalize(np.array(obs_buffer), norm.obs_mean, norm.obs_std)
        rtg_norm = normalize(np.array(rtg_buffer), norm.rtg_mean, norm.rtg_std)
#        obs_norm = norm.normalize_obs(np.array(obs_buffer))
#        rtg_norm = norm.normalize_rtg(np.array(rtg_buffer))

        actions_np = np.array(act_buffer)
        ts = np.array(timestep_buffer)

        obs_tensor = torch.tensor(obs_norm.copy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        act_tensor = torch.tensor(actions_np.copy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        rtg_tensor = torch.tensor(rtg_norm.copy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        ts_tensor  = torch.tensor(ts.copy(), dtype=torch.long).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # DTのMLP化検証 復元step8
            action_pred = model(ts_tensor, obs_tensor, act_tensor, rtg_tensor)
            action = action_pred[0, -1].cpu().numpy()

        # 実行
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        # バッファ更新
        obs_buffer.pop(0)
        obs_buffer.append(obs)

        act_buffer.pop(0)
        act_buffer.append(action)

        if not USE_FIXED_RTG:
            #最低２つないとpopで空になる
            if len(rtg_buffer) >= 2:
                rtg_buffer.pop(0)
                rtg_buffer.append(rtg_buffer[-1] - reward)  # 累積リターン更新

        timestep_buffer.pop(0)
        timestep_buffer.append(t % TIMESTEP_MAX)  # timestepは最大1024まで（Embedding制約）

        if done:
            print(f"✅ 終了ステップ数: {t}")
            print(f"✅ リプレイ情報の記録: start_waypoint_idx={env.scene.start_waypoint_idx} waypoint_direc={env.scene.waypoint_direc}")
            with open("replay_info.txt", "w") as f:
                 f.write(str(env.scene.start_waypoint_idx)+'\n')
                 f.write(str(env.scene.waypoint_direc)+'\n')
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
