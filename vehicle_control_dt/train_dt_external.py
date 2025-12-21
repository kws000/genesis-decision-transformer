import os
import torch
import torch.nn as nn
#import intel_npu_acceleration_library  # これでNPUバックエンドが登録される

from torch.utils.data import DataLoader
import argparse
import pickle

from model_dt import DecisionTransformer
from convert_to_dt_format import TIMESTEP_MAX
from train_dt import SequenceDataset  # または TrajectoryDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"

#計画と行動のマルチタスクモデル
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 2#30 #50 #※いまだけ削減
K_WP = 40
PLAN_M = 3
W_ACT = 1.0
W_PLAN = 0.5
W_SMOOTH = 0.01
USE_FOCUS = False
#EPOCHS = 30#50#100#20#100


#最新モデルでリプレイする　別手法
#TRY_CHECKPOINT_PATH = "checkpoints/temp_model.pt"
#TRY_PKL_PATH = "data_dt/trajectories_dt.pkl"
#TRY_NORM_PATH = "data_dt/mean_std.pkl"

def get_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None, -1
    steps = []
    for f in os.listdir(CHECKPOINT_DIR):
        if f.startswith("step") and f.endswith(".pt"):
            try:
                step_num = int(f[4:-3])
                steps.append((step_num, os.path.join(CHECKPOINT_DIR, f)))
            except:
                continue
    if not steps:
        return None, -1
    steps.sort()
    return steps[-1][1], steps[-1][0]

#計画と行動のマルチタスクモデル
def _zscore_nonneg(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean()
    sd = x.std(unbiased=False) + eps
    z = (x - mu) / sd
    return torch.clamp(z, min=0.0)


#計画と行動のマルチタスクモデル
def train_external(context_len, n_layer, n_head,norm_path,pkl_path,checkpoint_path,embed_dim=128):
#def train_external(context_len, n_layer, n_head,norm_path,pkl_path,checkpoint_path):

    # データ読み込み
    dataset = SequenceDataset(pkl_path, context_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    obs_dim = dataset[0][0].shape[-1]
    act_dim = dataset[0][1].shape[-1]

    # モデル定義

#計画と行動のマルチタスクモデル
    model = DecisionTransformer(obs_dim, act_dim,
                                context_len=context_len,
                                embed_dim=embed_dim,
                                n_layer=n_layer,
                                n_head=n_head,
                                plan_M=PLAN_M,
                                use_focus=USE_FOCUS).to(DEVICE)
#   model = DecisionTransformer(
#       obs_dim=obs_dim,
#       act_dim=act_dim,
#       context_len=context_len,
#       embed_dim=128,
#       n_layer=n_layer,
#       n_head=n_head,
#   ).to(DEVICE)


    # 前回モデルのロード試行
    prev_model_path, prev_step = get_latest_checkpoint()
    if prev_model_path:
        try:
            print(f"🔄 前回モデルをロード: {prev_model_path}")

            state_dict = torch.load(prev_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)

            print("✅ 前回モデルを引き継ぎました")
        except Exception as e:
            print(f"⚠️ 構造が異なるため、前のモデルは使用しません（{e}）")
            print("⚠️ 学習は新規開始されます。")
    else:
        print("⚠️ 前回ステップのモデルが存在しないので新規学習")


#計画と行動のマルチタスクモデル
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()
#   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#   loss_fn = nn.MSELoss()

    print("🚀 Training Start")

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
            #※planがないことを想定する必要はない            
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
#       for states, actions, returns, timesteps in dataloader:
#           states, actions, returns, timesteps = (
#               states.to(DEVICE),
#               actions.to(DEVICE),
#               returns.to(DEVICE),
#               timesteps.to(DEVICE),
#           )
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
#       print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.5f}")

    # 保存
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    step_id = prev_step + 1

    # ここで暫定モデル temp_model.pt が保存される

#最新モデルでリプレイする　別手法
    save_path = checkpoint_path
#    save_path = os.path.join(CHECKPOINT_DIR, f"temp_model.pt")

    torch.save(model.state_dict(), save_path)
    print(f"✅ 暫定モデルを保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_len", type=int, required=True)
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--n_head", type=int, required=True)
    #最新モデルでリプレイする　別手法
    parser.add_argument("--norm_path", type=str, required=True)
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    args = parser.parse_args()

    train_external(args.context_len, args.n_layer, args.n_head,
                   args.norm_path,args.pkl_path,args.checkpoint_path)
