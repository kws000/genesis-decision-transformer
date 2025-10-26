# visualize_dt_attention.py
# 例:
# python visualize_dt_attention.py \
#   --checkpoint checkpoints/step2.pt \
#   --pkl        checkpoints/step2_trajectories_dt.pkl \
#   --norm_path  checkpoints/step2_mean_std.pkl \
#   --context_len 1 --n_layer 2 --n_head 4 --embed_dim 128 \
#   --outdir viz/step2 \
#   --obs_names x,y,speed,yaw_sin,yaw_cos,heading_err,cte,next_wp_dx,next_wp_dy \
#   --sample_index 0

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn

# 必ず pyplot を import する前に
import os
os.environ["MPLBACKEND"] = "Agg"  # 予防的（なくてもOK）

import matplotlib
matplotlib.use("Agg")             # 非GUIバックエンド
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from model_dt import DecisionTransformer

# ★ あなたのローダを利用
from train_dt import SequenceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# パラメータ(model_dt.pyに合わせた)
TIMESTEP_MAX = 4000

# =============== Utils ===============
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--pkl",         type=str, required=True)
    p.add_argument("--norm_path",   type=str, default="")
    p.add_argument("--context_len", type=int, required=True)
    p.add_argument("--n_layer",     type=int, required=True)
    p.add_argument("--n_head",      type=int, required=True)
    p.add_argument("--embed_dim",   type=int, default=128)
    p.add_argument("--timestep_max",type=int, default=TIMESTEP_MAX)
    p.add_argument("--outdir",      type=str, default="viz")
    p.add_argument("--obs_names",   type=str, default="")
    p.add_argument("--sample_index",type=int, default=0)  # DataLoaderでこのサンプルを1件だけ取る
    return p.parse_args()


def plot_heatmap(M, title, outpath, xticks=None, yticks=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    if title: plt.title(title)
    if xticks is not None:
        plt.xticks(ticks=np.arange(len(xticks)), labels=xticks, rotation=90)
    if yticks is not None:
        plt.yticks(ticks=np.arange(len(yticks)), labels=yticks)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def split_heads(W, n_head):
    """
    W: (D, D) をヘッドごとに (n_head, d_head, D) へ分割（行方向の均等分割）。
    """
    D = W.shape[0]
    assert D % n_head == 0, f"D={D} not divisible by n_head={n_head}"
    d_head = D // n_head
    return W.reshape(n_head, d_head, D)


# =============== Main Visualization ===============
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Optional: obs_names
    obs_names = [s.strip() for s in args.obs_names.split(",")] if args.obs_names else None

    # Optional: normalization (for orig-scale sensitivity)
    std_vec = None
    if args.norm_path:
        try:
            with open(args.norm_path, "rb") as f:
                norm = pickle.load(f)
            # 例: {"obs_mean": np.array([...]), "obs_std": np.array([...])}
            if "obs_std" in norm:
                std_vec = np.asarray(norm["obs_std"])
        except Exception as e:
            print(f"⚠️ 正規化ファイルの読み込みに失敗: {e}")

    # === あなたのローダ（SequenceDataset）で読み込み ===
    dataset = SequenceDataset(args.pkl, args.context_len)
    if not (0 <= args.sample_index < len(dataset)):
        raise IndexError(f"sample_index {args.sample_index} is out of range (0..{len(dataset)-1})")
    subset = Subset(dataset, [args.sample_index])
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # (B=1, T=context_len, dims)
    states, actions, returns, timesteps = next(iter(loader))


    # ここで“どの窓か”を表示（CPUのまま .item() でOK）
    tau_start = int(timesteps[0, 0].item())
    tau_end   = int(timesteps[0, -1].item())
    print("=== アテンションマップの可視化 ===")
    print(f"Window timesteps (abs): {tau_start} → {tau_end}")

    states    = states.to(DEVICE)      # (1,T,obs_dim)
    actions   = actions.to(DEVICE)     # (1,T,act_dim)
    returns   = returns.to(DEVICE)     # (1,T,1)
    timesteps = timesteps.to(DEVICE)   # (1,T)

    obs_dim = states.shape[-1]
    act_dim = actions.shape[-1]

    # Build & load model
    model = DecisionTransformer(
        obs_dim=obs_dim, act_dim=act_dim,
        context_len=args.context_len, embed_dim=args.embed_dim,
        n_layer=args.n_layer, n_head=args.n_head, timestep_max=args.timestep_max
    ).to(DEVICE)
    sd = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ------------------ A) Runtime token-token attention ------------------
    model.enable_attention_hook(True)
    with torch.no_grad():
        _ = model(timesteps, states, actions, returns)
    attn_list = model.get_collected_attn()  # list of (B, heads, Lq, Lk) or (B,1,Lq,Lk)

    # Assume B=1 for visualization
    for li, attn in enumerate(attn_list):
        A = attn[0].numpy()  # (heads, Lq, Lk)
        # heads 次元が1の場合もループでOK
        for h in range(A.shape[0]):
            out = os.path.join(args.outdir, f"attn_layer{li}_head{h}.png")
            plot_heatmap(
                A[h],
                title=f"Attention L{li} H{h}",
                outpath=out,
                xticks=None, yticks=None
            )

    # ------------------ B) Static Wq/Wk/Wv × obs-dim sensitivity ---------
    qkv = model.get_qkv_weights()              # list[dict]
    W_state = model.get_state_embed_weight()   # (D, obs_dim)

    # For x-axis labels
    xticks = obs_names if (obs_names and len(obs_names) == obs_dim) else None

    for li, mats in enumerate(qkv):
        Wq = mats["W_q"].cpu().numpy()
        Wk = mats["W_k"].cpu().numpy()
        Wv = mats["W_v"].cpu().numpy()

        Wq_h = split_heads(torch.from_numpy(Wq), args.n_head).numpy()  # (H, d_head, D)
        Wk_h = split_heads(torch.from_numpy(Wk), args.n_head).numpy()
        Wv_h = split_heads(torch.from_numpy(Wv), args.n_head).numpy()
        Ws   = W_state.cpu().numpy()  # (D, obs_dim)

        q_score, k_score, v_score = [], [], []
        for h in range(args.n_head):
            Mq = Wq_h[h] @ Ws  # (d_head, obs_dim)
            Mk = Wk_h[h] @ Ws
            Mv = Wv_h[h] @ Ws
            q_score.append(np.linalg.norm(Mq, axis=0))  # (obs_dim,)
            k_score.append(np.linalg.norm(Mk, axis=0))
            v_score.append(np.linalg.norm(Mv, axis=0))

        q_score = np.stack(q_score, axis=0)  # (H, obs_dim)
        k_score = np.stack(k_score, axis=0)
        v_score = np.stack(v_score, axis=0)

        plot_heatmap(q_score, f"Wq×W_state (Layer {li})",
                     os.path.join(args.outdir, f"Wq_obs_L{li}.png"),
                     xticks=xticks, yticks=[f"h{h}" for h in range(args.n_head)])
        plot_heatmap(k_score, f"Wk×W_state (Layer {li})",
                     os.path.join(args.outdir, f"Wk_obs_L{li}.png"),
                     xticks=xticks, yticks=[f"h{h}" for h in range(args.n_head)])
        plot_heatmap(v_score, f"Wv×W_state (Layer {li})",
                     os.path.join(args.outdir, f"Wv_obs_L{li}.png"),
                     xticks=xticks, yticks=[f"h{h}" for h in range(args.n_head)])

        # 元スケール寄与（(x-mean)/std の std を補正）
        if std_vec is not None:
            eps = 1e-8
            inv_std = 1.0 / np.clip(std_vec, eps, None)  # (obs_dim,)
            q_score_orig = q_score * inv_std
            k_score_orig = k_score * inv_std
            v_score_orig = v_score * inv_std

            plot_heatmap(q_score_orig, f"Wq×W_state (orig-scale) L{li}",
                         os.path.join(args.outdir, f"Wq_obs_L{li}_origscale.png"),
                         xticks=xticks, yticks=[f"h{h}" for h in range(args.n_head)])
            plot_heatmap(k_score_orig, f"Wk×W_state (orig-scale) L{li}",
                         os.path.join(args.outdir, f"Wk_obs_L{li}_origscale.png"),
                         xticks=xticks, yticks=[f"h{h}" for h in range(args.n_head)])
            plot_heatmap(v_score_orig, f"Wv×W_state (orig-scale) L{li}",
                         os.path.join(args.outdir, f"Wv_obs_L{li}_origscale.png"),
                         xticks=xticks, yticks=[f"h{h}" for h in range(args.n_head)])

    print("✅ 可視化を出力:", os.path.abspath(args.outdir))
    print("   - attn_layer{L}_head{H}.png : トークン間アテンション (3T×3T)")
    print("   - Wq_obs_L{L}.png / Wk_obs_L{L}.png / Wv_obs_L{L}.png : 観測次元×ヘッド感度")
    if std_vec is not None:
        print("   - *_origscale.png : 正規化を元単位に寄せた感度表示（std補正済み）")


if __name__ == "__main__":
    main()
