import numpy as np, pandas as pd, pickle, os

#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 経過確認ログ
from schema import OBS_COLS, ACT_COLS, PLAN_COLS, REWARD_COL

def probe_csv(path="expert_data/expert_data.csv"):
    import pandas as pd
    print(f"[PROBE/CSV] load: {path}")
    df = pd.read_csv(path)
    print("[PROBE/CSV] rows:", len(df), "cols:", list(df.columns))

    must = OBS_COLS + ACT_COLS + PLAN_COLS + [REWARD_COL]
    missing = [c for c in must if c not in df.columns]
    if missing:
        raise RuntimeError(f"[PROBE/CSV] missing cols: {missing}")

    for c in must:
        if df[c].isna().any():
            raise RuntimeError(f"[PROBE/CSV] NaN in col: {c}")

    th = df["throttle"].astype(float)
    st = df["steer_angle"].astype(float)
    rw = df[REWARD_COL].astype(float)
    print(f"[PROBE/CSV] throttle mean={th.mean():.4f} q90={th.quantile(0.9):.4f} min={th.min():.4f} max={th.max():.4f}")
    print(f"[PROBE/CSV] steer    mean={st.mean():.4f} q90={st.quantile(0.9):.4f} min={st.min():.4f} max={st.max():.4f}")
    print(f"[PROBE/CSV] reward   mean={rw.mean():.4f} q90={rw.quantile(0.9):.4f} min={rw.min():.4f} max={rw.max():.4f}")
    print("[PROBE/CSV] ok.")


#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 経過確認ログ
def probe_raw_pkl(path="trajectories/trajectory_data.pkl"):
    print(f"[PROBE/RAW] load: {path}")
    with open(path,"rb") as f:
        trajs = pickle.load(f)
    if not trajs: raise RuntimeError("[PROBE/RAW] empty list")
    t0 = trajs[0]
    print("[PROBE/RAW] keys:", list(t0.keys()))
    for k in ["obs","action","reward","done","next_obs","plan"]:
        a = t0.get(k,None)
        if a is None: raise RuntimeError(f"[PROBE/RAW] missing {k}")
        print(f"[PROBE/RAW] {k} shape:", np.shape(a))
        if np.isnan(a).any(): raise RuntimeError(f"[PROBE/RAW] NaN in {k}")
    print("[PROBE/RAW] ok.")

#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 経過確認ログ
def probe_dt(path_data="data_dt/trajectories_dt.pkl", path_norm="data_dt/mean_std.pkl"):
    print(f"[PROBE/DT] load: {path_data}")
    with open(path_data,"rb") as f:
        D = pickle.load(f)
    t0 = D[0]
    print("[PROBE/DT] keys:", list(t0.keys()))
    for k in ["observations","actions","returns","timesteps","initial_rtg"]:
        a = t0.get(k,None)
        if a is None: raise RuntimeError(f"[PROBE/DT] missing {k}")
        print(f"[PROBE/DT] {k} shape:", np.shape(a))
    has_plan = "plan" in t0
    if has_plan:
        print("[PROBE/DT] plan shape:", np.shape(t0["plan"]))

    print(f"[PROBE/DT] norm: {path_norm}")
    with open(path_norm,"rb") as f:
        N = pickle.load(f)
    for k in ["obs_mean","obs_std","ret_mean","ret_std"]:
        if k not in N: raise RuntimeError(f"[PROBE/DT] missing {k} in norm")
    obs_std = np.asarray(N["obs_std"])
    ones_like = np.isclose(obs_std, 1.0, atol=1e-6).sum()
    print(f"[PROBE/DT] obs_dim={len(N['obs_mean'])}  std==1.0 columns={ones_like}")
    if ones_like > 0:
        print("[PROBE/DT] ⚠ 1.0固定のstdが残っています（正規化ロジックを確認）")
    print("[PROBE/DT] ok.")
