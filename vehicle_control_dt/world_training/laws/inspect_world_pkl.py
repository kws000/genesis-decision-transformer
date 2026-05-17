import os
import glob
import pickle
import argparse
import numpy as np


# ============================================================
# OBS keys fixed in current project
# ============================================================

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


# ============================================================
# Helpers
# ============================================================

def safe_stats(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p01": float(np.percentile(x, 1)),
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
    }


def print_stats(name, x):
    s = safe_stats(x)

    print(f"\n{name}")
    print(
        f"mean={s['mean']:.6f}  "
        f"std={s['std']:.6f}  "
        f"min={s['min']:.6f}  "
        f"max={s['max']:.6f}"
    )
    print(
        f"p01={s['p01']:.6f}  "
        f"p10={s['p10']:.6f}  "
        f"p50={s['p50']:.6f}  "
        f"p90={s['p90']:.6f}  "
        f"p99={s['p99']:.6f}"
    )


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_mean_std(path):
    obj = load_pickle(path)

    # 想定候補に広めに対応
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
            raise RuntimeError(
                f"Unknown mean/std dict keys: {list(obj.keys())}"
            )
    elif isinstance(obj, (tuple, list)) and len(obj) >= 2:
        obs_mean = obj[0]
        obs_std = obj[1]
    else:
        raise RuntimeError(
            f"Unsupported mean/std format: {type(obj)}"
        )

    obs_mean = np.asarray(obs_mean, dtype=np.float32).reshape(-1)
    obs_std = np.asarray(obs_std, dtype=np.float32).reshape(-1)

    if obs_mean.shape[0] != len(OBS_V2_KEYS):
        raise RuntimeError(
            f"obs_mean dim mismatch: {obs_mean.shape[0]} != {len(OBS_V2_KEYS)}"
        )

    if obs_std.shape[0] != len(OBS_V2_KEYS):
        raise RuntimeError(
            f"obs_std dim mismatch: {obs_std.shape[0]} != {len(OBS_V2_KEYS)}"
        )

    obs_std = np.where(np.abs(obs_std) < 1e-8, 1.0, obs_std)

    return obs_mean, obs_std

#損失に世界モデルを使う　可視化　#テンソル構造が逆？
def denorm_obs(obs_norm, obs_mean, obs_std):
    obs_norm = np.asarray(obs_norm, dtype=np.float32)

    # (1, 19) や (1, 1, 19) で来ても (19,) に潰す
    obs_norm = obs_norm.reshape(-1)

    if obs_norm.shape[0] != 19:
        raise RuntimeError(f"obs_norm must flatten to 19 dims, got {obs_norm.shape}")

    return obs_norm * obs_std + obs_mean
#def denorm_obs(obs_norm, obs_mean, obs_std):
#    return obs_norm * obs_std[None, :] + obs_mean[None, :]

def require_key(item, key, traj_idx, ds_name):
    if key not in item:
        raise RuntimeError(
            f"[{ds_name} traj={traj_idx}] Missing key: {key}. "
            f"Available keys: {list(item.keys())}"
        )
    return item[key]


# ============================================================
# Main inspection
# ============================================================

def inspect_dataset(ds_dir, obs_mean, obs_std):
    ds_name = os.path.basename(ds_dir)

    pkl_path = os.path.join(ds_dir, "trajectories_dt.pkl")

    print("\n")
    print("=" * 90)
    print(ds_name)
    print("=" * 90)

    if not os.path.exists(pkl_path):
        print(f"[WARN] Missing trajectories_dt.pkl: {pkl_path}")
        return

    data = load_pickle(pkl_path)

    if not isinstance(data, list):
        raise RuntimeError(
            f"[{ds_name}] trajectories_dt.pkl should be list, got {type(data)}"
        )

    print(f"[INFO] trajectories: {len(data)}")

    idx_vel = OBS_V2_KEYS.index("velocity")
    idx_perp = OBS_V2_KEYS.index("perp_error")
    idx_heading = OBS_V2_KEYS.index("heading_error")
    idx_vratio = OBS_V2_KEYS.index("v_ratio")
    idx_limit_v = OBS_V2_KEYS.index("limit_v_target")

    total_steps = 0
    total_pairs = 0
    outside_phys_count = 0

    for traj_idx, item in enumerate(data):
        obs_norm = require_key(item, "observations", traj_idx, ds_name)
        actions = require_key(item, "actions", traj_idx, ds_name)

        obs_norm = np.asarray(obs_norm, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if obs_norm.ndim != 2:
            raise RuntimeError(f"observations must be 2D, got {obs_norm.shape}")

        if actions.ndim != 2:
            raise RuntimeError(f"actions must be 2D, got {actions.shape}")

        T, obs_dim = obs_norm.shape
        _, act_dim = actions.shape

        if obs_dim != len(OBS_V2_KEYS):
            raise RuntimeError(
                f"obs_dim mismatch: {obs_dim} != {len(OBS_V2_KEYS)}"
            )

        if actions.shape[0] != T:
            raise RuntimeError(
                f"actions length mismatch: {actions.shape[0]} != {T}"
            )

        obs_phys = denorm_obs(obs_norm, obs_mean, obs_std)

        total_steps += T
        total_pairs += max(T - 1, 0)

        print("\n--- trajectory info ---")
        print(f"traj_idx          : {traj_idx}")
        print(f"observations shape: {obs_norm.shape}  # normalized")
        print(f"actions shape     : {actions.shape}")

        if "returns" in item:
            print(f"returns shape     : {np.asarray(item['returns']).shape}")
        if "timesteps" in item:
            print(f"timesteps shape   : {np.asarray(item['timesteps']).shape}")
        if "plan" in item:
            print(f"plan shape        : {np.asarray(item['plan']).shape}")
        if "wp_preview" in item:
            print(f"wp_preview shape  : {np.asarray(item['wp_preview']).shape}")

        print("\nobs keys:")
        for i, k in enumerate(OBS_V2_KEYS):
            print(f"{i:2d}: {k}")

        if not np.isfinite(obs_norm).all():
            raise RuntimeError(f"[{ds_name} traj={traj_idx}] obs_norm has NaN/Inf")

        if not np.isfinite(actions).all():
            raise RuntimeError(f"[{ds_name} traj={traj_idx}] actions has NaN/Inf")

        if not np.isfinite(obs_phys).all():
            raise RuntimeError(f"[{ds_name} traj={traj_idx}] obs_phys has NaN/Inf")

        # ----------------------------------------------------
        # Physical-space diagnostics
        # ----------------------------------------------------
        print("\n[PHYS OBS] denormalized by base_mean_std.pkl")

        print_stats("perp_error_phys", obs_phys[:, idx_perp])
        outside = np.abs(obs_phys[:, idx_perp]) > 1.0
        outside_rate = float(np.mean(outside))
        outside_count = int(np.sum(outside))
        outside_phys_count += outside_count

        print(
            f"outside_rate_phys(abs(perp)>1.0): "
            f"{outside_rate:.6f}  count={outside_count}/{T}"
        )

        print_stats("heading_error_phys", obs_phys[:, idx_heading])
        print_stats("velocity_phys", obs_phys[:, idx_vel])
        print_stats("v_ratio_phys", obs_phys[:, idx_vratio])
        print_stats("limit_v_target_phys", obs_phys[:, idx_limit_v])

        # ----------------------------------------------------
        # Normalized-space diagnostics
        # ----------------------------------------------------
        print("\n[NORM OBS] observations in trajectories_dt.pkl")

        print_stats("perp_error_norm", obs_norm[:, idx_perp])
        print_stats("heading_error_norm", obs_norm[:, idx_heading])
        print_stats("velocity_norm", obs_norm[:, idx_vel])
        print_stats("v_ratio_norm", obs_norm[:, idx_vratio])

        # ----------------------------------------------------
        # Delta diagnostics
        # World Model target候補: obs_norm[t+1] - obs_norm[t]
        # ----------------------------------------------------
        if T >= 2:
            delta_norm = obs_norm[1:] - obs_norm[:-1]
            delta_phys = obs_phys[1:] - obs_phys[:-1]

            print("\n[DELTA NORM] target for World MLP")
            print_stats("Δperp_error_norm", delta_norm[:, idx_perp])
            print_stats("Δheading_error_norm", delta_norm[:, idx_heading])
            print_stats("Δvelocity_norm", delta_norm[:, idx_vel])
            print_stats("Δv_ratio_norm", delta_norm[:, idx_vratio])

            print("\n[DELTA PHYS] reference only")
            print_stats("Δperp_error_phys", delta_phys[:, idx_perp])
            print_stats("Δheading_error_phys", delta_phys[:, idx_heading])
            print_stats("Δvelocity_phys", delta_phys[:, idx_vel])
            print_stats("Δv_ratio_phys", delta_phys[:, idx_vratio])

        # ----------------------------------------------------
        # Action diagnostics
        # ----------------------------------------------------
        print("\n[ACTION]")
        if act_dim >= 1:
            print_stats("steer/action[:,0]", actions[:, 0])
        if act_dim >= 2:
            print_stats("throttle/action[:,1]", actions[:, 1])

    print("\n")
    print("-" * 90)
    print(f"total_steps              : {total_steps}")
    print(f"total_world_model_pairs  : {total_pairs}")
    print(f"outside_phys_count       : {outside_phys_count}")
    print("-" * 90)


def main():
    parser = argparse.ArgumentParser()

    default_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data"
        )
    )

    default_mean_std = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data_dt",
            "base_mean_std.pkl"
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_root,
        help="world_training/data directory containing ds_* folders",
    )

    parser.add_argument(
        "--mean_std",
        type=str,
        default=default_mean_std,
        help="Path to base_mean_std.pkl",
    )

    args = parser.parse_args()

    print("=" * 90)
    print("[inspect_world_pkl.py]")
    print(f"data_dir : {args.data_dir}")
    print(f"mean_std : {args.mean_std}")
    print("=" * 90)

    if not os.path.exists(args.mean_std):
        raise FileNotFoundError(
            f"mean_std not found: {args.mean_std}\n"
            f"Pass explicitly, for example:\n"
            f"python inspect_world_pkl.py "
            f"--mean_std C:\\Users\\kws00\\Genesis4D\\my_projects\\genesis-decision-transformer\\vehicle_control_dt\\data_dt\\base_mean_std.pkl"
        )

    obs_mean, obs_std = load_mean_std(args.mean_std)

    print("\n[mean/std loaded]")
    print(f"obs_mean shape: {obs_mean.shape}")
    print(f"obs_std shape : {obs_std.shape}")

    ds_dirs = sorted(glob.glob(os.path.join(args.data_dir, "ds_*")))

    print("\n" + "=" * 90)
    print(f"[INFO] Found ds folders: {len(ds_dirs)}")
    print("=" * 90)

    if len(ds_dirs) == 0:
        print(f"[ERROR] No ds_* found under: {args.data_dir}")
        return

    for ds_dir in ds_dirs:
        inspect_dataset(ds_dir, obs_mean, obs_std)


if __name__ == "__main__":
    main()