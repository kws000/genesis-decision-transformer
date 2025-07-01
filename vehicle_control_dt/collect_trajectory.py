from genesis_gym_env import GenesisEnv
from utils.trajectory_utils import save_trajectory

from stable_baselines3 import SAC


import torch
#import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 事前学習済みのポリシー（例: ControlMLP または SACモデル）

# 模倣学習から生成(ControlMLP)

# === 行動クローンモデルの読み込み ===
class ControlMLP(torch.nn.Module):
    
    # ９次元に拡張し順序を合わせる→１０次元へ
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
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

sac_model = None
bc_model = None

mode = 1#0 # 0:SAC 1:ControlMLP

if mode == 0:
    # 強化学習から生成(SAC)
    from stable_baselines3 import SAC
    sac_model = SAC.load("checkpoints/sac_genesis_20000_steps.zip")
else:
    # 模倣学習から生成(ControlMLP)
    bc_model = ControlMLP()
    bc_model.load_state_dict(torch.load("models/bc_model.pth"))
    bc_model.eval()



env = GenesisEnv()
obs = env.reset()


# 学習済みモデルによる走行


observations = []
actions = []
rewards = []
dones = []
next_observations = []
#trajectory = []

for t in range(10000):

    action = None

    if mode == 0:
        # SAC時
        if sac_model is not None:
            action, _ = sac_model.predict(obs, deterministic=True)
    else:
        # ControlMLP時
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        if bc_model is not None:
            action_tensor = bc_model(obs_tensor)
            action = action_tensor.squeeze(0).cpu().detach().numpy()

    next_obs, reward, done, info = env.step(action)
    
    # 軌跡を記録
    observations.append(obs)
    actions.append(action)  # ← これで Tステップ分のアクションを記録
    rewards.append(reward)
    dones.append(done)
    next_observations.append(next_obs)

#    trajectory.append({
#        'obs': obs,
#        'action': action,
#        'reward': reward,
#        'next_obs': next_obs,
#        'done': done
#    })

    obs = next_obs

    if done:
        break

env.close()

# 軌跡を保存

trajectory = {
    "obs": observations,
    "action": actions,
    "reward": rewards,
    "done": dones,
    "next_obs": next_observations,
}

save_trajectory([trajectory], "trajectories/trajectory_data.pkl")
#                  ↑ ここが重要

print("✅ 軌跡を trajectories/trajectory_data.pkl に保存しました")
