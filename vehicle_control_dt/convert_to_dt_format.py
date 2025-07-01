import pickle
import numpy as np
import os

# パス設定
input_path = "trajectories/trajectory_data.pkl"
output_path = "data_dt/trajectories_dt.pkl"
norm_path = "data_dt/mean_std.pkl"
os.makedirs("trajectories", exist_ok=True)
os.makedirs("data_dt", exist_ok=True)

# パラメータ
TIMESTEP_MAX = 4000

# データ読み込み
with open(input_path, "rb") as f:
    raw_trajectories = pickle.load(f)

# 軌跡ごとのデータ抽出
observations, actions, returns = [], [], []


print("type of raw_trajectories:", type(raw_trajectories))
if isinstance(raw_trajectories, dict):
    print("Keys:", raw_trajectories.keys())

discounted_return_last = []

for traj in raw_trajectories:
    obs = traj["obs"]
    act = traj["action"]
    rew = traj["reward"]


    print("type:", type(traj["action"]))
    print("len:", len(traj["action"]))
    print("element 0:", traj["action"][0])
    print("element 0 shape:", np.shape(traj["action"][0]))



    # 報酬がスカラーならリストに

# 警告抑制
    if isinstance(rew, (float, np.floating)):
        rew = [rew]
#    if isinstance(rew, (float, np.float32, np.float64)):
#        rew = [rew]

    # RTG 計算
    discounted_return = []
    ret = 0
    for r in reversed(rew):
        ret += r
        discounted_return.insert(0, ret)

    # reshape により (T,) → (T, 1) を明示
    obs = np.array(obs)

    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)

    act = np.array(act)
    if act.ndim == 1:
        act = act.reshape(-1, 1)

    discounted_return = np.array(discounted_return)
    if discounted_return.ndim == 1:
        discounted_return = discounted_return.reshape(-1, 1)

    discounted_return_last = discounted_return

    observations.append(obs)
    
    actions.append(act)
    returns.append(discounted_return)

    print("action sample shape:", actions[0].shape)

# 正規化用統合
all_obs = np.concatenate(observations, axis=0)
all_returns = np.concatenate(returns, axis=0)

# === 統計量の更新 or 読み込み ===
if os.path.exists(norm_path):
    with open(norm_path, "rb") as f:
        norm_data = pickle.load(f)
    obs_mean_prev = norm_data["obs_mean"]
    obs_std_prev = norm_data["obs_std"]
    count_prev = norm_data.get("count", 1)

    if obs_mean_prev.shape[0] != all_obs.shape[1]:
        print("⚠️ 観測次元が前回と異なります。統計をリセットします。")
        obs_mean_prev = np.zeros(all_obs.shape[1], dtype=np.float32)
        obs_std_prev = np.ones(all_obs.shape[1], dtype=np.float32)
        count_prev = 0
else:
    obs_mean_prev = np.zeros(all_obs.shape[1], dtype=np.float32)
    obs_std_prev = np.ones(all_obs.shape[1], dtype=np.float32)
    count_prev = 0

count_new = all_obs.shape[0]
obs_mean_new = all_obs.mean(axis=0)
obs_std_new = all_obs.std(axis=0) + 1e-6

obs_mean = (obs_mean_prev * count_prev + obs_mean_new * count_new) / (count_prev + count_new)
obs_std = np.maximum(obs_std_prev, obs_std_new)
ret_mean = all_returns.mean(axis=0)
ret_std = all_returns.std(axis=0) + 1e-6

# 保存用リスト
dt_trajectories = []
for obs, act, ret in zip(observations, actions, returns):
    obs_norm = (obs - obs_mean) / obs_std
    ret_norm = (ret - ret_mean) / ret_std
    timesteps = np.arange(len(obs), dtype=np.int64).reshape(-1)

    # TIMESTEP_MAX 以上はまずい
    timesteps = timesteps % TIMESTEP_MAX

    dt_trajectories.append({
        "observations": obs_norm,
        "actions": act,
        "returns": ret_norm,
        "timesteps": timesteps,
        "initial_rtg": discounted_return_last,  # ←追加        
    })

print("✅ 変換されたエピソード数:", len(dt_trajectories))

# --- 保存 ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(dt_trajectories, f)

with open(norm_path, "wb") as f:
    pickle.dump({
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "ret_mean": ret_mean,
        "ret_std": ret_std,
        "count": count_prev + count_new,
    }, f)

print("✅ DTフォーマット変換および統計保存が完了しました")
