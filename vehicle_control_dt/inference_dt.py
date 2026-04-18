import torch
import pickle
import numpy as np

from model_dt import DecisionTransformer
from utils.trajectory_utils import normalize
from genesis_gym_env import GenesisEnv

import matplotlib.pyplot as plt
import seaborn as sns

from model_dt import visualize_attention

# === パラメータ ===

# 本番モード（Trueにしないとやがて崩壊する）
USE_FIXED_RTG = True
# 時間最大
TIMESTEP_MAX = 4000

#進化ループの大改修	正規化の固定統計
BASE_NORM_PKL = "data_dt/base_mean_std.pkl"   # ★固定統計

## DTのMLP化検証
#context_len = 1

## DTのMLP化検証 復元step1
#context_len = 1

## DTのMLP化検証 復元step2
#context_len = 5

## DTのMLP化検証 復元step3
#context_len = 5

## DTのMLP化検証 復元step4
#context_len = 3

## DTのMLP化検証 復元step5
#context_len = 3

## DTのMLP化検証 復元step6
#context_len = 3

## DTのMLP化検証 復元step7
#context_len = 20#3

# DTのMLP化検証 復元step8
context_len = 1#5#20

## DTのMLP化検証 復元
#context_len = 20


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "models/decision_transformer.pt"
pkl_path = "data_dt/trajectories_dt.pkl"

# === 統計情報の読み込み ===
with open(BASE_NORM_PKL, "rb") as f:
    stats = pickle.load(f)
obs_mean, obs_std = stats["obs_mean"], stats["obs_std"]
rtg_mean, rtg_std = stats["ret_mean"], stats["ret_std"]

# === DTデータ読み込み（1件目から形式確認用）
with open(pkl_path, "rb") as f:
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
print("obs_mean, obs_std:", obs_mean, obs_std)
print("rtg_mean, rtg_std:", rtg_mean, rtg_std)

# === モデル復元 ===

## DTのMLP化検証
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1]
#).to(device)

## DTのMLP化検証 復元step1
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1],
#).to(device)

## DTのMLP化検証 復元step2
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1],
#).to(device)

## DTのMLP化検証 復元step3
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1],
#    context_len=context_len
#).to(device)

## DTのMLP化検証 復元step4
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1],
#    context_len=context_len
#).to(device)

## DTのMLP化検証 復元step5
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1],
#    context_len=context_len
#).to(device)

## DTのMLP化検証 復元step6
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1],
#    context_len=context_len
#).to(device)

## DTのMLP化検証 復元step7
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1],
#    context_len=context_len
#).to(device)

# DTのMLP化検証 復元step8
model = DecisionTransformer(
    obs_dim=traj["observations"].shape[1],
    act_dim=traj["actions"].shape[1],
    context_len=context_len
).to(device)

## DTのMLP化検証 復元
#model = DecisionTransformer(
#    obs_dim=traj["observations"].shape[1],
#    act_dim=traj["actions"].shape[1],
#    context_len=context_len
#).to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

## === 環境初期化 ===
env = GenesisEnv()
obs = env.reset()

## DTのMLP化検証
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

## DTのMLP化検証 復元step1
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

## DTのMLP化検証 復元step2
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

## DTのMLP化検証 復元step3
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

## DTのMLP化検証 復元step4
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

## DTのMLP化検証 復元step5
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

## DTのMLP化検証 復元step6
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

## DTのMLP化検証 復元step7
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

# DTのMLP化検証 復元step8
obs_buffer = [obs] * context_len
act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
timestep_buffer = [0] * context_len  # これは使われなくなるが残してOK

## DTのMLP化検証 復元
#obs_buffer = [obs] * context_len
#act_buffer = [np.zeros_like(traj["actions"][0])] * context_len
#timestep_buffer = [0] * context_len

## DTのMLP化検証 復元step5
#initial_rtg = np.array([0.0], dtype=np.float32)  # shape: (1,)

## DTのMLP化検証 復元step6
#initial_rtg = np.array([0.0], dtype=np.float32)  # shape: (1,)

## DTのMLP化検証 復元step7
#initial_rtg = np.array([0.0], dtype=np.float32)  # shape: (1,)

# DTのMLP化検証 復元step8
initial_rtg = np.array([1.0], dtype=np.float32)  # shape: (1,)

## DTのMLP化検証 復元
#initial_rtg = np.array([0.0], dtype=np.float32)  # shape: (1,)
#initial_rtg = np.array([1000.0], dtype=np.float32)  # shape: (1,)


rtg_buffer = [initial_rtg.copy() for _ in range(context_len)]

for t in range(100_000):
    # 正規化＋テンソル化
    obs_norm = normalize(np.array(obs_buffer), obs_mean, obs_std)

#    print("obs before norm:", obs_buffer[-1])
#    print("obs after norm :", obs_norm[-1])

    rtg_norm = normalize(np.array(rtg_buffer), rtg_mean, rtg_std)
    actions_np = np.array(act_buffer)
    ts = np.array(timestep_buffer)

    obs_tensor     = torch.tensor(obs_norm.copy(), dtype=torch.float32).unsqueeze(0).to(device)
    act_tensor = torch.tensor(actions_np.copy(), dtype=torch.float32).unsqueeze(0).to(device)
    rtg_tensor = torch.tensor(rtg_norm.copy(), dtype=torch.float32).unsqueeze(0).to(device)
    ts_tensor  = torch.tensor(ts.copy(), dtype=torch.long).unsqueeze(0).to(device)

#    print("shape of obs_tensor:", np.shape(obs_tensor))
#    print("shape of act_tensor:", np.shape(act_tensor))
#    print("shape of rtg_tensor:", np.shape(rtg_tensor))
#    print("shape of ts_tensor:", np.shape(ts_tensor))

#    obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0).to(device)
#    act_tensor = torch.tensor(actions_np, dtype=torch.float32).unsqueeze(0).to(device)
#    rtg_tensor = torch.tensor(rtg_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
#    ts_tensor = torch.tensor(ts, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():

## DTのMLP化検証
#        action_pred = model(None, obs_tensor)  # returns_to_go など渡さない
#        action = action_pred[0, -1].cpu().numpy()

## DTのMLP化検証 復元step1
#        action_pred = model(None, obs_tensor)  # returns_to_go など渡さない
#        action = action_pred[0, -1].cpu().numpy()

## DTのMLP化検証 復元step2
#        action_pred = model(None, obs_tensor)  # returns_to_go など渡さない
#        action = action_pred[0, -1].cpu().numpy()

## DTのMLP化検証 復元step3
#        action_pred = model(ts_tensor, obs_tensor)  # returns_to_go など渡さない
#        action = action_pred[0, -1].cpu().numpy()

## DTのMLP化検証 復元step4
#        action_pred = model(ts_tensor, obs_tensor, act_tensor)  # returns_to_go など渡さない
#        action = action_pred[0, -1].cpu().numpy()

## DTのMLP化検証 復元step5
#        action_pred = model(ts_tensor, obs_tensor, act_tensor, rtg_tensor)
#        action = action_pred[0, -1].cpu().numpy()

## DTのMLP化検証 復元step6
#        action_pred = model(ts_tensor, obs_tensor, act_tensor, rtg_tensor)
#        action = action_pred[0, -1].cpu().numpy()

## DTのMLP化検証 復元step7
#        action_pred = model(ts_tensor, obs_tensor, act_tensor, rtg_tensor)
#        action = action_pred[0, -1].cpu().numpy()

# DTのMLP化検証 復元step8
        action_pred = model(ts_tensor, obs_tensor, act_tensor, rtg_tensor)
        action = action_pred[0, -1].cpu().numpy()

    # 実行
    obs, reward, done, _ = env.step(action)

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
        break

env.close()
