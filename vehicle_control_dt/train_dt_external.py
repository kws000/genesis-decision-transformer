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

#損失に世界モデルを使う
from world_training.laws.world_model_runtime import (
    WorldModelRuntime,
    default_paths_from_project,
)

#損失に世界モデルを使う 損失とつなぐ
import torch.nn.functional as F

try:
    from world_training.laws.world_mlp import WorldMLP
except ModuleNotFoundError:
    from world_training.laws.world_mlp import WorldMLP

#学習の重さの理由？無効化してみる
os.environ["MPLBACKEND"] = "Agg"  # これが一番確実


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

#損失に世界モデルを使う
WORLD_TRAINING_DIR = "world_training"

# ====== hyper (keep yours) ======
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5#50#30
K_WP = 40
PLAN_M = 3

#各種損失の係数

#損失に世界モデルを使う ステートをフィードバック更新する スロットとステア損失を分解
W_ACT_STEER = 0.0      # 診断用
W_ACT_THROTTLE = 1.0   # 維持
#W_ACT = 0.0#1.0

W_PLAN = 0.5
W_SMOOTH = 0.01
W_SM = 0.25

LOW_SPEED = 5.0
W_LOW_SPEED = 0.1

#低速罰の不具合
# 低速時に最低限ほしいアクセル
THROTTLE_MIN_LOW_SPEED = 0.20


USE_FOCUS = False
FORCE_CLIP = 3.0

#進化ループの大改修
STEPS_PER_EPOCH = 2000#50#※多すぎるのであとで調整
#STEPS_PER_EPOCH = 2000

NUM_SAMPLES = BATCH_SIZE * STEPS_PER_EPOCH  # 128_000

		
#損失に世界モデルを使う 損失とつなぐ
IDX_PERP = 7
IDX_HEAD = 8

#損失に世界モデルを使う ステートをフィードバック更新する
USE_WM_ROLLOUT = True

#損失に世界モデルを使う ステートをフィードバック更新する Nan対策
LAMBDA_WM_ROLLOUT_ACT = 0.009#0.006#0.003#0.005#0.003
WM_BLEND_ALPHA = 0.3#0.20
WM_ROLLOUT_H = 1
#LAMBDA_WM_ROLLOUT_ACT = 0.01
#WM_ROLLOUT_H = 2
#WM_BLEND_ALPHA = 0.30


USE_WORLD_LOSS = True
LAMBDA_WM = 0.03#0.01

WM_INPUT_DIM = 23
WM_OUTPUT_DIM = 19
WM_HIDDEN_DIM = 128

# 損失に世界モデルを使う ステートをフィードバック更新する
def compute_wm_rollout_action_loss(
    model,
    wm,
    timesteps,
    states,
    actions,
    returns,
    pred_actions,
    wp=None,
    rollout_h=2,
    alpha=0.3,
):
    """
    WMでperp/headだけ更新した観測列を作り、
    その観測でDTを再推論し、teacher actionへ寄せる。

    現在は診断モード:
      - blended_states は作る
      - DT再推論は torch.no_grad()
      - そのため L_wm_rollout は学習勾配には効かない
      - NaNが消えるか確認するための段階
    """
    if wm is None:
        return states.new_tensor(0.0), states

    blended_states = build_wm_blended_states(
        wm=wm,
        states=states,
        actions_gt=actions,
        pred_actions=pred_actions.detach(),
        rollout_h=rollout_h,
        alpha=alpha,
    )

    # blended state 自体の安全確認
    if not torch.isfinite(blended_states).all():
        print("[WARN] blended_states has NaN/Inf. skip L_wm_rollout.")
        return states.new_tensor(0.0), blended_states

    # 値域が大きすぎる場合も一旦スキップ
    max_abs_state = blended_states.detach().abs().max().item()
    if max_abs_state > 50.0:
        print(f"[WARN] blended_states too large max_abs={max_abs_state:.3f}. skip L_wm_rollout.")
        return states.new_tensor(0.0), blended_states

#損失に世界モデルを使う ステートをフィードバック更新する Nan対策　
    out = model(
        timesteps=timesteps,
        states=blended_states,
        actions=actions,
        returns=returns,
        wp=wp,
        return_plan=True,
        return_focus=USE_FOCUS,
    )
    pred_actions_blended = out[0] if isinstance(out, tuple) else out
#    # 診断モード: ここでは勾配を流さない
#    with torch.no_grad():
#        out = model(
#            timesteps=timesteps,
#            states=blended_states,
#            actions=actions,
#            returns=returns,
#            wp=wp,
#            return_plan=True,
#            return_focus=USE_FOCUS,
#        pred_actions_blended = out[0] if isinstance(out, tuple) else out

    if not torch.isfinite(pred_actions_blended).all():
        print("[WARN] pred_actions_blended has NaN/Inf. skip L_wm_rollout.")
        return states.new_tensor(0.0), blended_states

    pred_actions_blended = torch.nan_to_num(
        pred_actions_blended,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    H = min(rollout_h, states.shape[1] - 1)

    if H <= 0:
        return states.new_tensor(0.0), blended_states

    diff = pred_actions_blended[:, 1:H + 1] - actions[:, 1:H + 1]
    diff = torch.clamp(diff, -2.0, 2.0)

    loss = (diff ** 2).mean()

    #損失に世界モデルを使う ステートをフィードバック更新する Nan対策
    perp_max = blended_states[..., IDX_PERP].abs().max().item()
    head_max = blended_states[..., IDX_HEAD].abs().max().item()
    print(
        "[WM_ROLL_DEBUG] "
        f"loss={loss.item():.8f} "
        f"perp_max={perp_max:.4f} "
        f"head_max={head_max:.4f} "
        f"pred_blend_max={pred_actions_blended.detach().abs().max().item():.4f}"
    )

    if not torch.isfinite(loss):
        print("[WARN] L_wm_rollout became NaN/Inf. replaced with 0.")
        loss = states.new_tensor(0.0)

    # 診断モードではlossを返すが、no_grad由来なので勾配は流れない
    return loss, blended_states

#損失に世界モデルを使う ステートをフィードバック更新する
def build_wm_blended_states(
    wm,
    states,
    actions_gt,
    pred_actions,
    rollout_h=2,
    alpha=0.3,
):
    """
    Step3B:
      states の perp/head だけを World Model 結果で順次更新する。
      ただし teacher next と blend する。

    states:
      (B,T,19) normalized teacher obs

    actions_gt:
      (B,T,2) teacher actions

    pred_actions:
      (B,T,2) DT predicted actions

    return:
      blended_states: (B,T,19)
    """
    if wm is None:
        return states

    B, T, obs_dim = states.shape

    if T < 2:
        return states

    H = min(rollout_h, T - 1)

    blended = states.clone()

    # 初期stateは教師
    state_roll = states[:, 0].clone()

    prev_action = torch.zeros_like(actions_gt[:, 0])

    for t in range(H):
        pred_action_t = pred_actions[:, t]

        wm_x = torch.cat(
            [
                state_roll,
                prev_action,
                pred_action_t,
            ],
            dim=-1,
        )  # (B,23)

        delta = wm(wm_x)  # (B,19)
        wm_next = state_roll + delta

        #損失に世界モデルを使う ステートをフィードバック更新する nan対策

        # NaN/Inf guard
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        # perp/head の更新量だけを安全範囲に制限
        delta_perp = torch.clamp(delta[:, IDX_PERP], -0.5, 0.5)
        delta_head = torch.clamp(delta[:, IDX_HEAD], -0.3, 0.3)

        wm_next = state_roll.clone()
        wm_next[:, IDX_PERP] = state_roll[:, IDX_PERP] + delta_perp
        wm_next[:, IDX_HEAD] = state_roll[:, IDX_HEAD] + delta_head



        teacher_next = states[:, t + 1]

        next_state = teacher_next.clone()

        # perp/head だけ World Model と teacher を blend
        next_state[:, IDX_PERP] = (
            alpha * wm_next[:, IDX_PERP]
            + (1.0 - alpha) * teacher_next[:, IDX_PERP]
        )

        #損失に世界モデルを使う ステートをフィードバック更新する clamp
        next_state[:, IDX_PERP] = torch.clamp(
            next_state[:, IDX_PERP],
            -5.0,
            5.0,
        )

        next_state[:, IDX_HEAD] = (
            alpha * wm_next[:, IDX_HEAD]
            + (1.0 - alpha) * teacher_next[:, IDX_HEAD]
        )

        #損失に世界モデルを使う ステートをフィードバック更新する clamp
        next_state[:, IDX_HEAD] = torch.clamp(
            next_state[:, IDX_HEAD],
            -3.0,
            3.0,
        )

        blended[:, t + 1] = next_state

        # 次step入力へ伝播
        state_roll = next_state

        # prev_action は実際に使った pred_action を渡す
        prev_action = pred_action_t

    return blended

#損失に世界モデルを使う 損失とつなぐ
def compute_world_reflex_loss(
    wm,
    states,
    actions_gt,
    pred_actions,
):
    """
    states:
        (B,T,19) normalized obs

    actions_gt:
        (B,T,2) teacher actions
        prev_action作成用

    pred_actions:
        (B,T,2) DT predicted actions

    return:
        scalar loss
    """
    if wm is None:
        return states.new_tensor(0.0)

    B, T, obs_dim = states.shape

    if T < 1:
        return states.new_tensor(0.0)

    # prev_action[t]
    prev_actions = torch.zeros_like(actions_gt)
    if T > 1:
        prev_actions[:, 1:] = actions_gt[:, :-1]

    # World Model input:
    # obs_t + prev_action_t + pred_action_t
    wm_x = torch.cat(
        [
            states,
            prev_actions,
            pred_actions,
        ],
        dim=-1,
    )  # (B,T,23)

    wm_x_flat = wm_x.reshape(B * T, -1)

    delta_next_flat = wm(wm_x_flat)
    delta_next = delta_next_flat.reshape(B, T, obs_dim)

    wm_next = states + delta_next

    perp_now = states[..., IDX_PERP]
    head_now = states[..., IDX_HEAD]

    perp_next = wm_next[..., IDX_PERP]
    head_next = wm_next[..., IDX_HEAD]

    # 悪化した分だけ罰する
    loss_perp = F.relu(torch.abs(perp_next) - torch.abs(perp_now)).mean()
    loss_head = F.relu(torch.abs(head_next) - torch.abs(head_now)).mean()

    loss_wm = loss_perp + loss_head

    return loss_wm

#損失に世界モデルを使う 損失とつなぐ
def load_frozen_world_model(project_root, device):
    wm_path = os.path.join(
        project_root,
        "world_training",
        "models",
        "world_mlp.pt",
    )

    if not os.path.exists(wm_path):
        print(f"[WM] not found: {wm_path}")
        return None

    ckpt = torch.load(
        wm_path,
        map_location=device,
        weights_only=False,
    )

    wm = WorldMLP(
        input_dim=ckpt.get("input_dim", WM_INPUT_DIM),
        output_dim=ckpt.get("output_dim", WM_OUTPUT_DIM),
        hidden_dim=ckpt.get("hidden_dim", WM_HIDDEN_DIM),
    ).to(device)

    wm.load_state_dict(ckpt["model_state_dict"])
    wm.eval()

    for p in wm.parameters():
        p.requires_grad_(False)

    print(f"[WM] loaded frozen world model: {wm_path}")

    return wm


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

    #損失に世界モデルを使う
    project_root = os.path.dirname(os.path.abspath(__file__))

    model_path, mean_std_path = default_paths_from_project(project_root)

    wm_runtime = WorldModelRuntime(
        model_path=model_path,
        mean_std_path=mean_std_path,
    )

	#損失に世界モデルを使う 損失とつなぐ
    wm_model = None
    if USE_WORLD_LOSS:
        wm_model = load_frozen_world_model(project_root, DEVICE)


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

#損失に世界モデルを使う ステートをフィードバック更新する スロットとステア損失を分解
            act_diff = (pred_actions - actions) ** 2
            L_act_steer = (w[..., 0] * act_diff[..., 0]).mean()
            L_act_throttle = (w[..., 1] * act_diff[..., 1]).mean()
            L_act = (
                W_ACT_STEER * L_act_steer
                + W_ACT_THROTTLE * L_act_throttle
            )
#           L_act = (w * (pred_actions - actions) ** 2).mean()


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

#低速罰の不具合
            # 速度が低速なのにアクセルが小さい場合に罰する
            low_speed_delta = torch.clamp(LOW_SPEED - vel_phys, min=0.0)
            low_speed_mask = (low_speed_delta > 0.0).float()
            # pred_th が 0.20 未満なら連続的に罰する
            low_throttle_penalty = F.relu(THROTTLE_MIN_LOW_SPEED - pred_th) ** 2
            low_speed_denom = low_speed_mask.sum() + 1e-6
            L_low_speed = (low_throttle_penalty * low_speed_mask).sum() / low_speed_denom
#            # 速度が低速なのにアクセルを踏まないのも損失
#            low_speed_delta = torch.clamp(LOW_SPEED - vel_phys, min=0.0)
#            low_speed_mask = (low_speed_delta > 0).float()
#            low_speed_denom = low_speed_mask.sum() + 1e-6
#            L_low_speed = (( pred_th <= 0.05) * low_speed_mask ).sum() / low_speed_denom

#損失に世界モデルを使う ステートをフィードバック更新する
            L_wm_rollout = states.new_tensor(0.0)

            if USE_WM_ROLLOUT and wm_model is not None:
                L_wm_rollout, blended_states = compute_wm_rollout_action_loss(
                    model=model,
                    wm=wm_model,
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns=returns,
                    pred_actions=pred_actions,
                    wp=wp if "wp" in locals() else None,
                    rollout_h=WM_ROLLOUT_H,
                    alpha=WM_BLEND_ALPHA,
                )

            loss = (
#損失に世界モデルを使う ステートをフィードバック更新する スロットとステア損失を分解
                L_act              
#               W_ACT * L_act
                + W_PLAN * L_plan
                + W_SMOOTH * L_smooth
                + W_SM * L_sm
                + W_LOW_SPEED * L_low_speed
                + LAMBDA_WM_ROLLOUT_ACT * L_wm_rollout
            )

#			#損失に世界モデルを使う 損失とつなぐ
#            L_wm = compute_world_reflex_loss(
#                wm=wm_model,
#                states=states,
#                actions_gt=actions,
#                pred_actions=pred_actions,
#            )
#
#           loss = W_ACT * L_act + W_PLAN * L_plan + W_SMOOTH * L_smooth + W_SM * L_sm + W_LOW_SPEED * L_low_speed + LAMBDA_WM * L_wm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            avg_loss = total_loss / max(1, len(dataloader))

            counter += 1

            print(
                f"Epoch {epoch+1:03d} ({counter}/{len(dataloader)}) | L={avg_loss:.5f} "
#損失に世界モデルを使う ステートをフィードバック更新する スロットとステア損失を分解
                f"| L_act_s={L_act_steer.item():.4f} "
                f"| L_act_t={L_act_throttle.item():.4f} "
#                f"| L_act={L_act.item():.4f} "
                f"| L_plan={L_plan.item():.4f} "
#損失に世界モデルを使う ステートをフィードバック更新する                
                f"| L_wm_roll={L_wm_rollout.item():.6f} "                
 #               f"| L_wm={L_wm.item():.6f} "
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
