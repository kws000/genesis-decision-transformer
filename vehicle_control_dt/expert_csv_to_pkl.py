#ボトルネック認識とVmax魂の注入 4.1 丸ごと置き換えた

# csv_to_pkl.py（collect_trajectory 形式に修正）

# export_csv_to_pkl.py  (collect_trajectory 形式 / OBS_V2 対応)

import pandas as pd
import pickle
import numpy as np
import os
import sys

CSV_PATH   = "expert_data/expert_data.csv"
OUTPUT_PKL = "trajectories/trajectory_data.pkl"
PLAN_M     = 3

# === OBS V2 (19次元) ===
OBS_V2_KEYS = [
    # 既存10
    "target_wp_relative_x","target_wp_relative_y","pos_x","pos_y",
    "yaw_sin","yaw_cos","velocity","perp_error","heading_error","passed",
    # 追加9（VMAX塊）
    "kappa_local","mu_local","vmax_local","v_ratio","headroom",
    "vmax_min_hH","vmax_mean_hH","vmax_slope_hH","limit_v_target",
]
ACT_KEYS  = ["steer_angle","throttle"]
PLAN_KEYS = [f"plan_x{i}" for i in range(1, PLAN_M+1)] + [f"plan_y{i}" for i in range(1, PLAN_M+1)]

def main():
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    #ボトルネック認識とVmax魂の注入 アクセルが負の数値になる
 #   print(df["throttle"].min(), df["throttle"].max())
 #   print((df["throttle"]<0).mean(), "← 負割合")
    print(df["throttle"].min(), df["throttle"].mean(), df["throttle"].quantile([0.5,0.9,0.99]), df["throttle"].max())


    # --- 必須列チェック ---
    missing = [c for c in OBS_V2_KEYS + ACT_KEYS + PLAN_KEYS + ["reward"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"[export_csv_to_pkl] Missing columns in CSV: {missing}")

    # --- obs (N,19) ---
    obs = df[OBS_V2_KEYS].to_numpy(np.float32)

    # --- act (N,2) ---
    act = df[ACT_KEYS].to_numpy(np.float32)

    # --- reward (N,) ---
    rew = df["reward"].to_numpy(np.float32)

    # --- plan (N,2*M) を [x1,y1,x2,y2,x3,y3] の順で作る ---
    plan = np.empty((len(df), 2*PLAN_M), dtype=np.float32)
    for i in range(PLAN_M):
        plan[:, 2*i + 0] = df[f"plan_x{i+1}"].to_numpy(np.float32)
        plan[:, 2*i + 1] = df[f"plan_y{i+1}"].to_numpy(np.float32)

    # --- next_obs / done ---
    next_obs = np.roll(obs, -1, axis=0)
    if len(obs) > 0:
        next_obs[-1] = obs[-1]
    done = np.zeros(len(obs), dtype=bool)
    if len(done) > 0:
        done[-1] = True

    # --- NaN/Inf ガード ---
    for name, arr in [("obs", obs), ("action", act), ("reward", rew), ("plan", plan)]:
        if not np.isfinite(arr).all():
            bad = np.where(~np.isfinite(arr))
            raise RuntimeError(f"[export_csv_to_pkl] {name} contains NaN/Inf at indices {bad}")

    # --- collect_trajectory 1本で保存 ---
    trajectory = {
        "obs": obs,            # (N,19)
        "action": act,         # (N,2)
        "reward": rew,         # (N,)
        "done": done,          # (N,)
        "next_obs": next_obs,  # (N,19)
        "plan": plan,          # (N, 2*M)
        # 参考メタ
        "obs_keys": OBS_V2_KEYS,
        "act_keys": ACT_KEYS,
        "plan_M": PLAN_M,
    }
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump([trajectory], f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved: {OUTPUT_PKL}")
    print(f"   N={len(obs)}, obs_dim={obs.shape[1]}, act_dim={act.shape[1]}, plan_dim={plan.shape[1]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[export_csv_to_pkl] ERROR: {e}", file=sys.stderr)
        raise
