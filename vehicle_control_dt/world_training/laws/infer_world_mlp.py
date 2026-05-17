import os
import glob
import pickle
import argparse
import numpy as np
import torch

try:
    from world_training.laws.world_mlp import WorldMLP
except ModuleNotFoundError:
    from world_mlp import WorldMLP

OBS_V2_KEYS = [
    "target_x", "target_y",
    "pos_x", "pos_y",
    "yaw_sin", "yaw_cos",
    "velocity",
    "perp_error",
    "heading_error",
    "passed",
    "kappa_local",
    "mu_local",
    "vmax_local",
    "v_ratio",
    "headroom",
    "vmax_min_hH",
    "vmax_mean_hH",
    "vmax_slope_hH",
    "limit_v_target",
]

KEYS_TO_SHOW = [
    "perp_error",
    "heading_error",
    "velocity",
    "v_ratio",
    "limit_v_target",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_mean_std(path):
    obj = load_pickle(path)

    if isinstance(obj, dict):
        if "obs_mean" in obj and "obs_std" in obj:
            obs_mean = obj["obs_mean"]
            obs_std = obj["obs_std"]
        elif "mean" in obj and "std" in obj:
            obs_mean = obj["mean"]
            obs_std = obj["std"]
        elif "state_mean" in obj and "state_std" in obj:
            obs_mean = obj["state_mean"]
            obs_std = obj["state_std"]
        else:
            raise RuntimeError(f"Unknown mean/std keys: {list(obj.keys())}")
    else:
        raise RuntimeError(f"Unsupported mean/std format: {type(obj)}")

    obs_mean = np.asarray(obs_mean, dtype=np.float32).reshape(-1)
    obs_std = np.asarray(obs_std, dtype=np.float32).reshape(-1)
    obs_std = np.where(np.abs(obs_std) < 1e-8, 1.0, obs_std)

    return obs_mean, obs_std

#エンバグで推論テストが死んでいた
def denorm_obs(obs_norm, obs_mean, obs_std):
    obs_norm = np.asarray(obs_norm, dtype=np.float32)

    # 単体: (19,)
    if obs_norm.ndim == 1:
        if obs_norm.shape[0] == 19:
            return obs_norm * obs_std + obs_mean

        # flattenされたcontext: (T*19,)
        if obs_norm.shape[0] > 19 and obs_norm.shape[0] % 19 == 0:
            obs_norm = obs_norm.reshape(-1, 19)[-1]
            return obs_norm * obs_std + obs_mean

        raise RuntimeError(f"obs_norm must be 19 dims or T*19 dims, got {obs_norm.shape}")

    # batch: (N,19)
    if obs_norm.ndim == 2:
        if obs_norm.shape[1] != 19:
            raise RuntimeError(f"obs_norm batch must be (N,19), got {obs_norm.shape}")
        return obs_norm * obs_std[None, :] + obs_mean[None, :]

    # context/batch混在: (...,19)
    if obs_norm.shape[-1] == 19:
        return obs_norm * obs_std + obs_mean

    raise RuntimeError(f"Unsupported obs_norm shape: {obs_norm.shape}")
#def denorm_obs(obs_norm, obs_mean, obs_std):
#    obs_norm = np.asarray(obs_norm, dtype=np.float32)
#
#    # (1, 19) や (1, 1, 19) で来ても (19,) に潰す
#    obs_norm = obs_norm.reshape(-1)
#
#    if obs_norm.shape[0] != 19:
#        raise RuntimeError(f"obs_norm must flatten to 19 dims, got {obs_norm.shape}")
#
#    return obs_norm * obs_std + obs_mean


def load_model(model_path, device):
    ckpt = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    input_dim = ckpt.get("input_dim", 23)
    output_dim = ckpt.get("output_dim", 19)
    hidden_dim = ckpt.get("hidden_dim", 128)

    model = WorldMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def predict_delta_norm(model, obs_norm, prev_action, action, device):
    x = np.concatenate(
        [
            obs_norm.astype(np.float32),
            prev_action.astype(np.float32),
            action.astype(np.float32),
        ],
        axis=0,
    )

    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    pred_delta = model(x_t).squeeze(0).cpu().numpy()

    return pred_delta


@torch.no_grad()
def predict_delta_norm_batch(model, obs_norm, prev_action, action, device, batch_size=4096):
    xs = np.concatenate(
        [
            obs_norm.astype(np.float32),
            prev_action.astype(np.float32),
            action.astype(np.float32),
        ],
        axis=1,
    )

    preds = []

    for s in range(0, len(xs), batch_size):
        xb = torch.from_numpy(xs[s:s + batch_size]).float().to(device)
        pb = model(xb).cpu().numpy()
        preds.append(pb)

    return np.concatenate(preds, axis=0)


def print_row(name, true_val, pred_val):
    diff = pred_val - true_val
    print(
        f"{name:18s} "
        f"true={true_val:+.6f}  "
        f"pred={pred_val:+.6f}  "
        f"diff={diff:+.6f}"
    )


def compare_one_step(
    model,
    obs_norm,
    actions,
    obs_mean,
    obs_std,
    idx,
    device,
):
    prev_action = np.zeros(2, dtype=np.float32)
    if idx > 0:
        prev_action = actions[idx - 1]

    action = actions[idx]
    teacher_next_norm = obs_norm[idx + 1]

    pred_delta_norm = predict_delta_norm(
        model=model,
        obs_norm=obs_norm[idx],
        prev_action=prev_action,
        action=action,
        device=device,
    )

    pred_next_norm = obs_norm[idx] + pred_delta_norm

    teacher_next_phys = denorm_obs(teacher_next_norm, obs_mean, obs_std)
    pred_next_phys = denorm_obs(pred_next_norm, obs_mean, obs_std)

    print("\n" + "=" * 80)
    print(f"[ONE STEP] idx={idx}")
    print("=" * 80)
    print(f"action      = {action}")
    print(f"prev_action = {prev_action}")

    for key in KEYS_TO_SHOW:
        k = OBS_V2_KEYS.index(key)
        print_row(key, teacher_next_phys[k], pred_next_phys[k])


def rollout_closed_loop(
    model,
    obs_norm,
    actions,
    obs_mean,
    obs_std,
    start_idx,
    horizon,
    device,
):
    state_norm = obs_norm[start_idx].copy()

    prev_action = np.zeros(2, dtype=np.float32)
    if start_idx > 0:
        prev_action = actions[start_idx - 1].copy()

    print("\n" + "=" * 80)
    print(f"[CLOSED LOOP] start_idx={start_idx}, horizon={horizon}")
    print("=" * 80)

    for h in range(horizon):
        idx = start_idx + h

        if idx >= len(actions) - 1:
            break

        action = actions[idx]

        pred_delta_norm = predict_delta_norm(
            model=model,
            obs_norm=state_norm,
            prev_action=prev_action,
            action=action,
            device=device,
        )

        state_norm = state_norm + pred_delta_norm

        teacher_next_norm = obs_norm[idx + 1]

        pred_phys = denorm_obs(state_norm, obs_mean, obs_std)
        teacher_phys = denorm_obs(teacher_next_norm, obs_mean, obs_std)

        print(f"\n--- h={h + 1}  idx={idx} -> {idx + 1} ---")
        print(f"action      = {action}")
        print(f"prev_action = {prev_action}")

        for key in KEYS_TO_SHOW:
            k = OBS_V2_KEYS.index(key)
            print_row(key, teacher_phys[k], pred_phys[k])

        prev_action = action.copy()


def summary_one_step(
    model,
    trajectories,
    obs_mean,
    obs_std,
    device,
    batch_size=4096,
):
    err_sums = {k: 0.0 for k in KEYS_TO_SHOW}
    abs_sums = {k: 0.0 for k in KEYS_TO_SHOW}
    count = 0

    for item in trajectories:
        obs_norm = np.asarray(item["observations"], dtype=np.float32)
        actions = np.asarray(item["actions"], dtype=np.float32)

        T = len(obs_norm)
        if T < 2:
            continue

        prev_actions = np.zeros_like(actions)
        prev_actions[1:] = actions[:-1]

        obs_t = obs_norm[:-1]
        act_t = actions[:-1]
        prev_act_t = prev_actions[:-1]
        true_next_norm = obs_norm[1:]

        pred_delta_norm = predict_delta_norm_batch(
            model=model,
            obs_norm=obs_t,
            prev_action=prev_act_t,
            action=act_t,
            device=device,
            batch_size=batch_size,
        )

        pred_next_norm = obs_t + pred_delta_norm

        true_next_phys = denorm_obs(true_next_norm, obs_mean, obs_std)
        pred_next_phys = denorm_obs(pred_next_norm, obs_mean, obs_std)

        diff = pred_next_phys - true_next_phys

        n = diff.shape[0]
        count += n

        for key in KEYS_TO_SHOW:
            k = OBS_V2_KEYS.index(key)
            d = diff[:, k]
            err_sums[key] += float(np.sum(d * d))
            abs_sums[key] += float(np.sum(np.abs(d)))

    print("\n" + "=" * 80)
    print("[SUMMARY: ONE STEP RMSE / MAE]")
    print("=" * 80)

    for key in KEYS_TO_SHOW:
        rmse = np.sqrt(err_sums[key] / max(count, 1))
        mae = abs_sums[key] / max(count, 1)
        print(f"{key:18s} rmse={rmse:.6f}  mae={mae:.6f}")

    print(f"\ncount={count}")


def summary_closed_loop(
    model,
    trajectories,
    obs_mean,
    obs_std,
    device,
    horizon=5,
):
    err_sums = {k: 0.0 for k in KEYS_TO_SHOW}
    abs_sums = {k: 0.0 for k in KEYS_TO_SHOW}
    count = 0

    for item in trajectories:
        obs_norm = np.asarray(item["observations"], dtype=np.float32)
        actions = np.asarray(item["actions"], dtype=np.float32)

        T = len(obs_norm)
        if T <= horizon:
            continue

        for start_idx in range(0, T - horizon):
            state_norm = obs_norm[start_idx].copy()

            prev_action = np.zeros(2, dtype=np.float32)
            if start_idx > 0:
                prev_action = actions[start_idx - 1].copy()

            for h in range(horizon):
                idx = start_idx + h
                action = actions[idx]

                pred_delta_norm = predict_delta_norm(
                    model=model,
                    obs_norm=state_norm,
                    prev_action=prev_action,
                    action=action,
                    device=device,
                )

                state_norm = state_norm + pred_delta_norm
                prev_action = action.copy()

            true_norm = obs_norm[start_idx + horizon]

            pred_phys = denorm_obs(state_norm, obs_mean, obs_std)
            true_phys = denorm_obs(true_norm, obs_mean, obs_std)

            diff = pred_phys - true_phys

            for key in KEYS_TO_SHOW:
                k = OBS_V2_KEYS.index(key)
                d = float(diff[k])
                err_sums[key] += d * d
                abs_sums[key] += abs(d)

            count += 1

    print("\n" + "=" * 80)
    print(f"[SUMMARY: CLOSED LOOP h={horizon} RMSE / MAE]")
    print("=" * 80)

    for key in KEYS_TO_SHOW:
        rmse = np.sqrt(err_sums[key] / max(count, 1))
        mae = abs_sums[key] / max(count, 1)
        print(f"{key:18s} rmse={rmse:.6f}  mae={mae:.6f}")

    print(f"\ncount={count}")


def load_trajectories(data_dir, ds):
    if ds == "all":
        ds_dirs = sorted(glob.glob(os.path.join(data_dir, "ds_*")))
    else:
        ds_dirs = [os.path.join(data_dir, ds)]

    all_items = []

    print("\n[LOAD DATASETS]")
    for ds_dir in ds_dirs:
        pkl_path = os.path.join(ds_dir, "trajectories_dt.pkl")

        if not os.path.exists(pkl_path):
            print(f"[WARN] missing: {pkl_path}")
            continue

        trajectories = load_pickle(pkl_path)
        print(f"{os.path.basename(ds_dir)}: {len(trajectories)} trajectories")

        for i, item in enumerate(trajectories):
            obs = np.asarray(item["observations"])
            act = np.asarray(item["actions"])
            print(
                f"  traj_idx={i}  observations={obs.shape}  actions={act.shape}"
            )

        all_items.extend(trajectories)

    if not all_items:
        raise RuntimeError(f"No trajectories loaded. data_dir={data_dir}, ds={ds}")

    return all_items


def main():
    parser = argparse.ArgumentParser()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(base_dir, "data"),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(base_dir, "models", "world_mlp.pt"),
    )

    parser.add_argument(
        "--mean_std",
        type=str,
        default=os.path.abspath(
            os.path.join(
                base_dir,
                "..",
                "data_dt",
                "base_mean_std.pkl",
            )
        ),
    )

    parser.add_argument("--ds", type=str, default="ds_000001")
    parser.add_argument("--traj_idx", type=int, default=0)
    parser.add_argument("--idx", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--summary_only", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] device   : {device}")
    print(f"[INFO] data_dir : {args.data_dir}")
    print(f"[INFO] ds       : {args.ds}")
    print(f"[INFO] model    : {args.model}")
    print(f"[INFO] mean_std : {args.mean_std}")

    obs_mean, obs_std = load_mean_std(args.mean_std)
    model = load_model(args.model, device)

    trajectories = load_trajectories(args.data_dir, args.ds)

    if args.summary or args.summary_only:
        summary_one_step(
            model=model,
            trajectories=trajectories,
            obs_mean=obs_mean,
            obs_std=obs_std,
            device=device,
        )

        summary_closed_loop(
            model=model,
            trajectories=trajectories,
            obs_mean=obs_mean,
            obs_std=obs_std,
            device=device,
            horizon=args.horizon,
        )

        if args.summary_only:
            return

    if args.traj_idx < 0 or args.traj_idx >= len(trajectories):
        raise RuntimeError(
            f"--traj_idx must be 0 <= traj_idx < {len(trajectories)}, "
            f"got {args.traj_idx}"
        )

    item = trajectories[args.traj_idx]

    obs_norm = np.asarray(item["observations"], dtype=np.float32)
    actions = np.asarray(item["actions"], dtype=np.float32)

    if args.idx < 0 or args.idx >= len(obs_norm) - 1:
        raise RuntimeError(
            f"--idx must be 0 <= idx < {len(obs_norm) - 1}, got {args.idx}"
        )

    compare_one_step(
        model=model,
        obs_norm=obs_norm,
        actions=actions,
        obs_mean=obs_mean,
        obs_std=obs_std,
        idx=args.idx,
        device=device,
    )

    rollout_closed_loop(
        model=model,
        obs_norm=obs_norm,
        actions=actions,
        obs_mean=obs_mean,
        obs_std=obs_std,
        start_idx=args.idx,
        horizon=args.horizon,
        device=device,
    )


if __name__ == "__main__":
    main()