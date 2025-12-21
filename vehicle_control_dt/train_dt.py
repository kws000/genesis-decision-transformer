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

#計画と行動のマルチタスクモデル
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
K_WP = 40
PLAN_M = 3
W_ACT = 1.0
W_PLAN = 0.5
W_SMOOTH = 0.01
USE_FOCUS = False
#BATCH_SIZE = 32
#EPOCHS = 100
#LR = 1e-3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

#計画と行動のマルチタスクモデル    
    def __init__(self, path, context_len):
        # 引数名の取り違え修正（pkl_path -> path）
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
#   def __init__(self, path, context_len):
#       with open(pkl_path, "rb") as f:
#            trajectories = pickle.load(f)

        self.context_len = context_len
        self.samples = []

#計画と行動のマルチタスクモデル    
        # データセット全体で固定のKを決める（バッチ結合のため）
        target_K = None
        for tr in trajectories:
            wpv = tr.get("wp_preview", None)
            if isinstance(wpv, np.ndarray):
                if wpv.ndim == 3:      # (T,K,5)
                    target_K = int(wpv.shape[1]); break
                elif wpv.ndim == 2:    # (K,5)
                    target_K = int(wpv.shape[0]); break
        if target_K is None:
            target_K = K_WP
        self.k_wp = target_K
        self.plan_dim = 2 * PLAN_M

        for traj in trajectories:
            obs = traj["observations"]
            act = traj["actions"]
            ret = traj["returns"]
            tms = traj["timesteps"]

#計画と行動のマルチタスクモデル 追加フィールド（あれば利用）
            wpv = traj.get("wp_preview", None)   # (T,K,5) or (K,5)
            plan = traj.get("plan", None)        # (2M,) or (T,2M)
#           initial_rtg = traj["initial_rtg"][0]


            T = len(obs)
            if T < context_len:
                continue

            for i in range(T - context_len):

                # 初期報酬を渡すように
                obs_seq = obs[i:i+context_len]
                act_seq = act[i:i+context_len]
                rtg_seq = ret[i:i+context_len]
                tms_seq = tms[i:i+context_len]

#計画と行動のマルチタスクモデル
                # --- WPプレフィクス（窓の末尾に合わせる） ---
                if isinstance(wpv, np.ndarray):
                    if wpv.ndim == 3:      # (T,K,5)
                        wp_k5 = wpv[i + context_len - 1]
                    elif wpv.ndim == 2:    # (K,5)
                        wp_k5 = wpv
                    else:
                        wp_k5 = np.zeros((self.k_wp, 5), dtype=np.float32)
                else:
                    wp_k5 = np.zeros((self.k_wp, 5), dtype=np.float32)
                # pad/trim to K
                if wp_k5.shape[0] < self.k_wp:
                    pad = np.zeros((self.k_wp - wp_k5.shape[0], 5), dtype=wp_k5.dtype)
                    wp_k5 = np.concatenate([wp_k5, pad], axis=0)
                elif wp_k5.shape[0] > self.k_wp:
                    wp_k5 = wp_k5[:self.k_wp]
                # --- 計画ラベル（なければゼロで占位） ---
                if isinstance(plan, np.ndarray):
                    if plan.ndim == 2:      # (T, 2M)
                        plan_vec = plan[i + context_len - 1]
                    else:                    # (2M,)
                        plan_vec = plan
                    # pad/trim to 2*PLAN_M
                    if plan_vec.shape[0] < self.plan_dim:
                        pad = np.zeros((self.plan_dim - plan_vec.shape[0],), dtype=np.float32)
                        plan_vec = np.concatenate([plan_vec.astype(np.float32), pad], axis=0)
                    elif plan_vec.shape[0] > self.plan_dim:
                        plan_vec = plan_vec[:self.plan_dim].astype(np.float32)
                else:
                    plan_vec = np.zeros((self.plan_dim,), dtype=np.float32)
                self.samples.append({
                    "states":    obs_seq,
                    "actions":   act_seq,
                    "returns":   rtg_seq,
                    "timesteps": tms_seq,
                    "wp":        wp_k5,      # (K,5)
                    "plan":      plan_vec,   # (2*PLAN_M,)
                })
#               self.samples.append({
#                   "states": obs_seq,
#                   "actions": act_seq,
#                   "returns": rtg_seq,
#                   "timesteps": tms_seq,
#               })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
#計画と行動のマルチタスクモデル
        rets = sample["returns"]
        if rets.ndim == 1:
            rets = rets[:, None]  # (T,) -> (T,1)
        return (
            torch.tensor(sample["states"],    dtype=torch.float32),  # (T, obs_dim)
            torch.tensor(sample["actions"],   dtype=torch.float32),  # (T, act_dim)
            torch.tensor(rets,                dtype=torch.float32),  # (T, 1)
            torch.tensor(sample["timesteps"], dtype=torch.long),     # (T,)
            torch.tensor(sample["wp"],        dtype=torch.float32),  # (K, 5)
            torch.tensor(sample["plan"],      dtype=torch.float32),  # (2*PLAN_M,)
        )
#       return (
#           torch.tensor(sample["states"], dtype=torch.float32),       # (context_len, obs_dim)
#           torch.tensor(sample["actions"], dtype=torch.float32),      # (context_len, act_dim)
#           torch.tensor(sample["returns"], dtype=torch.float32),      # (context_len, 1)
#           torch.tensor(sample["timesteps"], dtype=torch.long),       # (context_len,)
#       )



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



# --- 学習ループ ---

#計画と行動のマルチタスクモデル
def _zscore_nonneg(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean()
    sd = x.std(unbiased=False) + eps
    z = (x - mu) / sd
    return torch.clamp(z, min=0.0)


#計画と行動のマルチタスクモデル
def train(pkl_path, context_len, embed_dim=128, n_layer=2, n_head=4, model_path=None):
#def train():

    dataset = SequenceDataset(pkl_path, context_len)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    for i, (states, actions, returns, timesteps) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  states.shape = {states.shape}")
        print(f"  actions.shape = {actions.shape}")
        print(f"  returns.shape = {returns.shape}")
        print(f"  timesteps.shape = {timesteps.shape}")
        break  # 一旦1バッチだけ確認

#計画と行動のマルチタスクモデル
    sample = dataset[0]
    states0, actions0, _, _, wp0, plan0 = sample
    obs_dim = states0.shape[-1]
    act_dim = actions0.shape[-1]
#   obs_dim = dataset[0][0].shape[-1]
#   act_dim = dataset[0][1].shape[-1]



## DTのMLP化検証
#    model = DecisionTransformer(obs_dim, act_dim).to(DEVICE)

## DTのMLP化検証 復元step1
#    model = DecisionTransformer(obs_dim, act_dim).to(DEVICE)

## DTのMLP化検証 復元step2
#    model = DecisionTransformer(obs_dim, act_dim).to(DEVICE)

## DTのMLP化検証 復元step3
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(DEVICE)

## DTのMLP化検証 復元step4
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(DEVICE)

## DTのMLP化検証 復元step5
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(DEVICE)

## DTのMLP化検証 復元step6
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(DEVICE)

## DTのMLP化検証 復元step7
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len, embed_dim=embed_dim).to(DEVICE)

#計画と行動のマルチタスクモデル
    model = DecisionTransformer(obs_dim, act_dim,
                                context_len=context_len,
                                embed_dim=embed_dim,
                                n_layer=n_layer,
                                n_head=n_head,
                                plan_M=PLAN_M,
                                use_focus=USE_FOCUS).to(DEVICE)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ モデルの重みを読み込みました: {model_path}")
    else:
       print(f"⚠️ モデルが存在しないため、新規で学習を開始します")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()
#   model = DecisionTransformer(obs_dim, act_dim,context_len=context_len, embed_dim=embed_dim).to(DEVICE)
#
#   if os.path.exists(model_path):
#       model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#       print(f"✅ モデルの重みを読み込みました: {model_path}")
#   else:
#       print(f"⚠️ モデルが存在しないため、新規で学習を開始します")
#
#   optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#   loss_fn = nn.MSELoss()

    print("Start training...")

#計画と行動のマルチタスクモデル
    for epoch in range(EPOCHS):
        total_loss = 0
        for states, actions, returns, timesteps, wp, plan in dataloader:
            timesteps = timesteps.to(DEVICE).long()
            states    = states.to(DEVICE).float()
            actions   = actions.to(DEVICE).float()
            returns   = returns.to(DEVICE).float()
            wp        = wp.to(DEVICE).float()
            pred_actions, pred_plan, alpha = model(timesteps, states, actions, returns,
                                                    wp=wp, return_plan=True, return_focus=USE_FOCUS)
            # 行動損失（RTG重み付きBC）
            w = _zscore_nonneg(returns)  # (B,T,1)
            w = w.expand_as(actions)     # (B,T,2)
            L_act = (w * (pred_actions - actions) ** 2).mean()
            # 計画損失
            plan = plan.to(DEVICE).float()  # (B,2M)
            L_plan = mse(pred_plan, plan)
            # 滑らかさ損失（Δa）
            if pred_actions.shape[1] > 1:
                da = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
                L_smooth = (da ** 2).mean()
            else:
                L_smooth = torch.zeros((), device=DEVICE)
            loss = W_ACT * L_act + W_PLAN * L_plan + W_SMOOTH * L_smooth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            avg_loss = total_loss / max(1, len(dataloader))
            print(f"Epoch {epoch+1:03d} | L={avg_loss:.5f} | L_act={L_act.item():.4f} | L_plan={L_plan.item():.4f} | L_sm={L_smooth.item():.4f}")
#   for epoch in range(EPOCHS):
#       total_loss = 0
#       for states, actions, returns,timesteps in dataloader:
#           timesteps = timesteps.to(DEVICE)
#           states = states.to(DEVICE)
#           actions = actions.to(DEVICE)
#           returns = returns.to(DEVICE)
#
#           pred_actions = model(timesteps, states, actions, returns)
#           loss = loss_fn(pred_actions, actions)
#
#           optimizer.zero_grad()
#           loss.backward()
#           optimizer.step()
#
#           total_loss += loss.item()
#
#       avg_loss = total_loss / len(dataloader)
#       print(f"Epoch {epoch+1} - Loss: {avg_loss:.5f}")

    # 保存
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/decision_transformer.pt")
    print("Model saved to models/decision_transformer.pt")


if __name__ == "__main__":
    train(pkl_path, context_len,)
    
