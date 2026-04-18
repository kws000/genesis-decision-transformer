import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_dt import DecisionTransformer

# sampler
from train_dt import SnapshotBinMixerDataset

#学習の重さの理由？無効化してみる
os.environ["MPLBACKEND"] = "Agg"  # これが一番確実


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

# ====== hyper (keep yours) ======
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5#50#30
K_WP = 40
PLAN_M = 3

#各種損失の係数
W_ACT = 1.0
W_PLAN = 0.5
W_SMOOTH = 0.01
W_SM = 0.25

LOW_SPEED = 5.0
W_LOW_SPEED = 0.1

USE_FOCUS = False
FORCE_CLIP = 3.0

#進化ループの大改修
STEPS_PER_EPOCH = 2000#50#※多すぎるのであとで調整
#STEPS_PER_EPOCH = 2000

NUM_SAMPLES = BATCH_SIZE * STEPS_PER_EPOCH  # 128_000


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


def weight_from_returns(returns, floor=0.2):
    r = returns.detach()
    rmin = r.amin(dim=1, keepdim=True)
    rmax = r.amax(dim=1, keepdim=True)
    w = (r - rmin) / (rmax - rmin + 1e-6)     # [0,1]
    return floor + (1.0 - floor) * w          # [floor,1]


def load_norm_pkl(norm_path):
    with open(norm_path, "rb") as f:
        s = pickle.load(f)
    obs_mean = np.asarray(s["obs_mean"], np.float32)
    obs_std  = np.asarray(s["obs_std"],  np.float32)
    return obs_mean, obs_std


def train_external(context_len, n_layer, n_head, checkpoint_path, embed_dim=128):
    # =========================
    # dataset (snapshot×bin)
    # =========================
    train_dataset = SnapshotBinMixerDataset(
        root_dir="data_dt",
        context_len=context_len,
        num_samples=NUM_SAMPLES,
        seed=0,
    )

    # 最短：代表dsの norm を使う
    norm_path = train_dataset.get_norm_path_for_training()
    obs_mean_np, obs_std_np = load_norm_pkl(norm_path)
    obs_mean_t = torch.tensor(obs_mean_np, device=DEVICE).view(1, 1, -1)
    obs_std_t  = torch.tensor(obs_std_np,  device=DEVICE).view(1, 1, -1)

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,      # dataset内部で抽選する
        num_workers=0,      # stats取りたいので固定
        drop_last=True,
    )

    # infer dims
    obs_dim = train_dataset[0][0].shape[-1]
    act_dim = train_dataset[0][1].shape[-1]
    assert obs_dim == 19, f"obs_dim must be 19 (OBS_V2). got {obs_dim}"

    # =========================
    # model
    # =========================
    model = DecisionTransformer(
        obs_dim, act_dim,
        context_len=context_len,
        embed_dim=embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        plan_M=PLAN_M,
        force_clip=FORCE_CLIP,
        idle_throttle_init=0.0908,
        use_focus=USE_FOCUS
    ).to(DEVICE)

    def _probe_zero_once(tag: str):
        model.eval()
        with torch.no_grad():
            B, T = 1, 1
            t = torch.zeros(B, T, dtype=torch.long, device=DEVICE)
            s = torch.zeros(B, T, model.obs_dim, device=DEVICE)
            a = torch.zeros(B, T, model.act_dim, device=DEVICE)
            r = torch.zeros(B, T, 2, device=DEVICE)  # returns_vec
            pa, _, _ = model(t, s, a, r, wp=None, return_plan=False)
            print(f"[PROBE {tag}] zero-input pred =", pa[0, 0].detach().cpu().numpy())
            print(f"[PROBE {tag}] head.bias      =", model.predict_action.bias.detach().cpu().numpy())

    _probe_zero_once("before_ckpt")

    # load previous model if exists
    prev_model_path, prev_step = get_latest_checkpoint()
    if prev_model_path:
        try:
            print(f"🔄 前回モデルをロード: {prev_model_path}")
            state_dict = torch.load(prev_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            _probe_zero_once("after_ckpt")
            print("✅ 前回モデルを引き継ぎました")
        except Exception as e:
            print(f"⚠️ 構造が異なるため、前のモデルは使用しません（{e}）")
            print("⚠️ 学習は新規開始されます。")
    else:
        print("⚠️ 前回ステップのモデルが存在しないので新規学習")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    print("🚀 Training Start")

    print("len(dataset)    =", len(train_dataset))
    print("len(dataloader) =", len(dataloader), " batch_size=", BATCH_SIZE)

    for epoch in range(EPOCHS):
        total_loss = 0.0

        counter = 0

        for states, actions, returns, timesteps, wp, plan in dataloader:
            timesteps = timesteps.to(DEVICE).long()
            states    = states.to(DEVICE).float()
            actions   = actions.to(DEVICE).float()
            returns   = returns.to(DEVICE).float()  # (B,T,2)

#            # dataloader から取り出した直後（to(DEVICE)の前でOK）
#            print("plan shape:", plan.shape)
#            print("plan abs mean:", plan.abs().mean().item(), "max:", plan.abs().max().item())
#            print("plan nonzero ratio:", (plan.abs() > 1e-6).float().mean().item())

            # wp guard
            wp_in = None
            if wp is not None and hasattr(wp, "numel") and wp.numel() > 0:
                wp_in = wp.to(DEVICE).float()

            pred_actions, pred_plan, alpha = model(
                timesteps, states, actions, returns,
                wp=wp_in, return_plan=True, return_focus=USE_FOCUS
            )

            # ---- losses ----
            rtg_p = returns[..., 0:1]        # progress
            rtg_c = returns[..., 1:2]        # clean
            w_p = weight_from_returns(rtg_p, floor=0.2)          # (B,T,1)
            w_c = torch.clamp(rtg_c, 0.0, 1.0)
            w = (w_p * w_c).expand_as(actions)
            L_act = (w * (pred_actions - actions) ** 2).mean()

            if (pred_plan is None) or (plan is None):
                L_plan = torch.zeros((), device=DEVICE)
            else:
                plan = plan.to(DEVICE).float()
                L_plan = mse(pred_plan, plan)

            if pred_actions.shape[1] > 1:
                da = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
                L_smooth = (da ** 2).mean()
            else:
                L_smooth = torch.zeros((), device=DEVICE)

            # ---- speed margin (you currently 0.0 weight) ----
            states_phys = states * obs_std_t + obs_mean_t
            vel_phys  = states_phys[..., 6]
            vlim_phys = torch.clamp(states_phys[..., 18], min=0.05)
            delta = torch.clamp(vel_phys - vlim_phys, min=0.0)
            mask = (delta > 0.05).float()
            pred_th = pred_actions[..., 1]
            denom = mask.sum() + 1e-6

            # 速度超過しているのにアクセルを踏むのは損失
            L_sm = ((pred_th - 0.0) ** 2 * mask).sum() / denom

            # 速度が低速なのにアクセルを踏まないのも損失
            low_speed_delta = torch.clamp(LOW_SPEED - vel_phys, min=0.0)
            low_speed_mask = (low_speed_delta > 0).float()
            low_speed_denom = low_speed_mask.sum() + 1e-6
            L_low_speed = (( pred_th <= 0.05) * low_speed_mask ).sum() / low_speed_denom

            loss = W_ACT * L_act + W_PLAN * L_plan + W_SMOOTH * L_smooth + W_SM * L_sm + W_LOW_SPEED * L_low_speed

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            avg_loss = total_loss / max(1, len(dataloader))

            counter += 1

            print(
                f"Epoch {epoch+1:03d} ({counter}/{len(dataloader)}) | L={avg_loss:.5f} "
                f"| L_act={L_act.item():.4f} | L_plan={L_plan.item():.4f} "
                f"| L_smooth={L_smooth.item():.4f} | L_sm_cond={L_sm.item():.4f} | L_low_spd={L_low_speed.item():.4f} "
            )


        # ---- epoch end logs ----
        stats = train_dataset.get_and_reset_stats()
        total = max(1, stats["total"])

        by_ds = " ".join([f"{k}:{v/total:.2f}" for k, v in stats["by_ds"].items()])
        by_bin = " ".join([f"{k}:{v/total:.2f}" for k, v in stats["by_bin"].items()])

        print(f"Epoch {epoch+1:03d} | L={avg_loss:.5f} | sampler ds[{by_ds}] bin[{by_bin}]")

    # save
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ 暫定モデルを保存: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_len", type=int, required=True)
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--n_head", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, default=128)
    args = parser.parse_args()

    train_external(
        args.context_len, args.n_layer, args.n_head,
        checkpoint_path=args.checkpoint_path,
        embed_dim=args.embed_dim
    )
