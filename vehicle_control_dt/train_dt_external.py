import os
import torch
import torch.nn as nn
import intel_npu_acceleration_library  # これでNPUバックエンドが登録される

from torch.utils.data import DataLoader
import argparse
import pickle

from model_dt import DecisionTransformer
from convert_to_dt_format import TIMESTEP_MAX
from train_dt import SequenceDataset  # または TrajectoryDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints"
EPOCHS = 30#50#100#20#100

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

def train_external(context_len, n_layer, n_head,
                   norm_path,pkl_path,checkpoint_path):
    # データ読み込み
    dataset = SequenceDataset(pkl_path, context_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    obs_dim = dataset[0][0].shape[-1]
    act_dim = dataset[0][1].shape[-1]

    # モデル定義
    model = DecisionTransformer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        context_len=context_len,
        embed_dim=128,
        n_layer=n_layer,
        n_head=n_head,
    ).to(device)

    # 前回モデルのロード試行
    prev_model_path, prev_step = get_latest_checkpoint()
    if prev_model_path:
        try:
            print(f"🔄 前回モデルをロード: {prev_model_path}")
            state_dict = torch.load(prev_model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ 前回モデルを引き継ぎました")
        except Exception as e:
            print(f"⚠️ 構造が異なるため、前のモデルは使用しません（{e}）")
            print("⚠️ 学習は新規開始されます。")
    else:
        print("⚠️ 前回ステップのモデルが存在しないので新規学習")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("🚀 Training Start")
    for epoch in range(EPOCHS):
        total_loss = 0
        for states, actions, returns, timesteps in dataloader:
            states, actions, returns, timesteps = (
                states.to(device),
                actions.to(device),
                returns.to(device),
                timesteps.to(device),
            )

            pred_actions = model(timesteps, states, actions, returns)
            loss = loss_fn(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.5f}")

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
