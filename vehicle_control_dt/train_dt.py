# vehicle_control_dt/train_dt.py

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt

from model_dt import DecisionTransformer


# --- 設定 ---

TIMESTEP_MAX = 4000

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
context_len = 1#5

## DTのMLP化検証 復元
#context_len = 20

embed_dim = 128
# === ハイパーパラメータ ===

# DTのMLP化検証
BATCH_SIZE = 32
EPOCHS = 100#10#2
LR = 1e-3
#BATCH_SIZE_org = 64 # MLPだと 32
#EPOCHS_org = 20     # MLPだと 100
#LR_org = 1e-4       # MLPだと 1e-3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pkl_path = "data_dt/trajectories_dt.pkl"
model_path = "models/decision_transformer.pt"


# --- Dataset定義 ---
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

# 学習サンプリング方式の変更
class SequenceDataset(Dataset):
    def __init__(self, path, context_len):
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
        
        self.context_len = context_len
        self.samples = []

        for traj in trajectories:
            obs = traj["observations"]
            act = traj["actions"]
            ret = traj["returns"]
            tms = traj["timesteps"]
            initial_rtg = traj["initial_rtg"][0]
#            initial_rtg = traj.get("initial_rtg",ret[0])

            T = len(obs)
            if T < context_len:
                continue

            for i in range(T - context_len):

# 初期報酬を渡すように
                obs_seq = obs[i:i+context_len]
                act_seq = act[i:i+context_len]
                rtg_seq = ret[i:i+context_len]
                tms_seq = tms[i:i+context_len]

                self.samples.append({
                    "states": obs_seq,
                    "actions": act_seq,
                    "returns": rtg_seq,
                    "timesteps": tms_seq,
                })

#                self.samples.append({
#                    "states": obs[i:i+context_len],
#                    "actions": act[i:i+context_len],
#                    "returns": ret[i:i+context_len],
#                    "timesteps": tms[i:i+context_len],
#                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["states"], dtype=torch.float32),       # (context_len, obs_dim)
            torch.tensor(sample["actions"], dtype=torch.float32),      # (context_len, act_dim)
            torch.tensor(sample["returns"], dtype=torch.float32),      # (context_len, 1)
            torch.tensor(sample["timesteps"], dtype=torch.long),       # (context_len,)
        )

class TrajectoryDataset(Dataset):
    def __init__(self, path, context_len):
        with open(path, "rb") as f:
            trajectories = pickle.load(f)

        # 偏り可視化
        debug_actions = np.concatenate([traj["actions"] for traj in trajectories], axis=0)
        print("steering mean:", debug_actions[:, 0].mean())
        print("steering std :", debug_actions[:, 0].std())

        # 全アクションをまとめてプロット
        is_plot = False#True#False
        if is_plot:

            all_steerings = []

            for traj in trajectories:
                actions = traj["actions"]  # shape: (T, 2)
                steer = actions[:, 0]      # steering 成分だけ取り出す
                all_steerings.extend(steer.tolist())

            plt.hist(all_steerings, bins=50, alpha=0.7)
            plt.title("Steering Distribution")
            plt.xlabel("Steering Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

        self.obs = []
        self.actions = []
        self.returns = []
        self.timesteps = []

        for traj in trajectories:
            self.obs.append(traj["observations"])
            self.actions.append(traj["actions"])
            self.returns.append(traj["returns"])
            self.timesteps.append(traj["timesteps"])

        self.obs = np.array(self.obs)
        self.actions = np.array(self.actions)
        self.returns = np.array(self.returns)
        self.timesteps = np.array(self.timesteps)

        self.context_len = context_len
        self.length = len(self.obs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        o = self.obs[idx]
        a = self.actions[idx]
        r = self.returns[idx]
        t = self.timesteps[idx]
        # 👈 IndexError を防ぐ！
        t = t % TIMESTEP_MAX

        # (T,) → (T, 1) に変換（次元揃え）
        if o.ndim == 1:
            o = o[:, None]
        if a.ndim == 1:
            a = a[:, None]
        if r.ndim == 1:
            r = r[:, None]
        if t.ndim == 1:
            t = t[:, None]

        def pad_or_trim(x):
            T = x.shape[0]
            if T < self.context_len:
                pad = np.zeros((self.context_len - T, x.shape[1]), dtype=x.dtype)
                return np.concatenate([pad, x], axis=0)
            else:
                return x[-self.context_len:]

        o = pad_or_trim(o)
        a = pad_or_trim(a)
        r = pad_or_trim(r)
        t = pad_or_trim(t)

        return (
            torch.tensor(o, dtype=torch.float32),   # states
            torch.tensor(a, dtype=torch.float32),   # actions
            torch.tensor(r, dtype=torch.float32),   # returns
            torch.tensor(t, dtype=torch.long),      # timesteps
        )

# DTのMLP化検証 復元step8attn
def visualize_attention(attn_weights, title="Attention Map", layer=0, head=0):
    # attn_weights: list of [B, n_head, T, T]
    attn = attn_weights[layer][0, head]  # shape: (T, T)
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn.cpu().numpy(), cmap="viridis")
    plt.title(f"{title} - Layer {layer}, Head {head}")
    plt.xlabel("Key Token")
    plt.ylabel("Query Token")
    plt.show()


# --- 学習ループ ---
def train():

# 学習サンプリング方式の変更
    dataset = SequenceDataset(pkl_path, context_len)
#    dataset = TrajectoryDataset(pkl_path, context_len)

    # DTはシャッフルが必須
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    for i, (states, actions, returns, timesteps) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  states.shape = {states.shape}")
        print(f"  actions.shape = {actions.shape}")
        print(f"  returns.shape = {returns.shape}")
        print(f"  timesteps.shape = {timesteps.shape}")
        break  # 一旦1バッチだけ確認


    obs_dim = dataset[0][0].shape[-1]
    act_dim = dataset[0][1].shape[-1]

## DTのMLP化検証
#    model = DecisionTransformer(obs_dim, act_dim).to(device)

## DTのMLP化検証 復元step1
#    model = DecisionTransformer(obs_dim, act_dim).to(device)

## DTのMLP化検証 復元step2
#    model = DecisionTransformer(obs_dim, act_dim).to(device)

## DTのMLP化検証 復元step3
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(device)

## DTのMLP化検証 復元step4
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(device)

## DTのMLP化検証 復元step5
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(device)

## DTのMLP化検証 復元step6
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(device)

## DTのMLP化検証 復元step7
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len, embed_dim=embed_dim).to(device)

# DTのMLP化検証 復元step8
    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len, embed_dim=embed_dim).to(device)

## DTのMLP化検証 復元
#    model = DecisionTransformer(obs_dim, act_dim, context_len=context_len, embed_dim=embed_dim).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ モデルの重みを読み込みました: {model_path}")
    else:
        print(f"⚠️ モデルが存在しないため、新規で学習を開始します")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("Start training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for states, actions, returns,timesteps in dataloader:
            timesteps = timesteps.to(device)
            states = states.to(device)
            actions = actions.to(device)
            returns = returns.to(device)

            pred_actions = model(timesteps, states, actions, returns)
            loss = loss_fn(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.5f}")

    # 保存
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/decision_transformer.pt")
    print("Model saved to models/decision_transformer.pt")


if __name__ == "__main__":
    train()
