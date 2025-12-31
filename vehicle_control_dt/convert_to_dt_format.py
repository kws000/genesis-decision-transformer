
#ボトルネック認識とVmax魂の注入 5.1 丸ごと置き換えた

import os, pickle
import numpy as np

# ── パス設定 ─────────────────────────────────────────────
INPUT_PKL  = "trajectories/trajectory_data.pkl"   # export_csv_to_pkl の出力
OUTPUT_PKL = "data_dt/trajectories_dt.pkl"        # DT学習用
NORM_PKL   = "data_dt/mean_std.pkl"
os.makedirs("data_dt", exist_ok=True)

# ── ハイパラ ─────────────────────────────────────────────
TIMESTEP_MAX = 4096  # 位置埋め込みの語彙上限と揃える（例）

# ── OBS V2 スキーマ（参照用/検証用） ─────────────────────
OBS_V2_DIM = 19  # 既存10 + VMAX塊9

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def _finite_or_raise(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))
        raise RuntimeError(f"[convert_to_dt_format] {name} contains NaN/Inf at indices {bad}")

def main():
    # ── 入力ロード ───────────────────────────────────────
    with open(INPUT_PKL, "rb") as f:
        raw = pickle.load(f)

    # 形式: [trajectory, ...] 想定（collect_trajectory）
    if isinstance(raw, dict):
        # まれに dict1本のケースに保険
        raw = [raw]

    observations, actions, returns, plans = [], [], [], []
    have_plan = False

    for traj in raw:
        obs = _ensure_2d(np.asarray(traj["obs"], dtype=np.float32))          # (T, 19)
        act = _ensure_2d(np.asarray(traj["action"], dtype=np.float32))       # (T, A)
        rew = np.asarray(traj["reward"], dtype=np.float32).reshape(-1)       # (T,)
        pln = traj.get("plan", None)                                         # (T, 2M) or None

        # 形の基本検証
        assert obs.shape[1] == OBS_V2_DIM, f"obs_dim {obs.shape[1]} != {OBS_V2_DIM}"
        assert obs.shape[0] == act.shape[0] == rew.shape[0], "T mismatch among obs/act/reward"

        # RTG（割引なし累積） (T,1)
        rtg = np.zeros_like(rew, dtype=np.float32)
        acc = 0.0
        for i in range(len(rew) - 1, -1, -1):
            acc += rew[i]
            rtg[i] = acc
        rtg = rtg.reshape(-1, 1)

        # plan
        if pln is not None:
            pln = _ensure_2d(np.asarray(pln, dtype=np.float32))  # (T, 2M)
            assert pln.shape[0] == obs.shape[0], "T mismatch in plan"
            plans.append(pln)
            have_plan = True

        observations.append(obs)
        actions.append(act)
        returns.append(rtg)

    # ── 正規化統計 ──────────────────────────────────────

#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 原因改善
    all_obs  = np.concatenate(observations, axis=0).astype(np.float32)  # (∑T, 19)
    all_rtg  = np.concatenate(returns, axis=0).astype(np.float32)       # (∑T, 1)
    all_plan = np.concatenate(plans, axis=0).astype(np.float32) if have_plan else None
#    all_obs = np.concatenate(observations, axis=0)     # (∑T, 19)
#    all_rtg = np.concatenate(returns, axis=0)          # (∑T, 1)
#    all_plan = np.concatenate(plans, axis=0) if have_plan else None

    _finite_or_raise("observations", all_obs)
    _finite_or_raise("returns", all_rtg)
    if have_plan:
        _finite_or_raise("plan", all_plan)

#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 原因改善
#    # 既存統計の増分更新にも対応（なければリセット）
#    if os.path.exists(NORM_PKL):
#        with open(NORM_PKL, "rb") as f:
#            norm = pickle.load(f)
#        obs_mean_prev = np.asarray(norm["obs_mean"], dtype=np.float32)
#        obs_std_prev  = np.asarray(norm["obs_std"], dtype=np.float32)
#        count_prev    = int(norm.get("count", 0))
#        # 観測次元が変わったらリセット
#        if obs_mean_prev.shape[0] != all_obs.shape[1]:
#            obs_mean_prev = np.zeros(all_obs.shape[1], dtype=np.float32)
#            obs_std_prev  = np.ones(all_obs.shape[1], dtype=np.float32)
#            count_prev    = 0
#    else:
#        obs_mean_prev = np.zeros(all_obs.shape[1], dtype=np.float32)
#        obs_std_prev  = np.ones(all_obs.shape[1], dtype=np.float32)
#        count_prev    = 0
#
#    count_new = all_obs.shape[0]

#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 原因改善
    obs_mean = all_obs.mean(axis=0).astype(np.float32)
    obs_std  = all_obs.std(axis=0).astype(np.float32) + 1e-6
#    obs_mean_new = all_obs.mean(axis=0)
#    obs_std_new  = all_obs.std(axis=0) + 1e-6
#
#    # 合成（分布変動に強くするため std は最大値を採用）
#    obs_mean = (obs_mean_prev * count_prev + obs_mean_new * count_new) / max(1, (count_prev + count_new))
#    obs_std  = np.maximum(obs_std_prev, obs_std_new)

    # RTGはバッチ正規化用途：ここでは全体で標準化（1次元）

#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 原因改善
    ret_mean = all_rtg.mean(axis=0).astype(np.float32)   # shape (1,)
    ret_std  = all_rtg.std(axis=0).astype(np.float32) + 1e-6
#    ret_mean = all_rtg.mean(axis=0)
#    ret_std  = all_rtg.std(axis=0) + 1e-6

    if have_plan:
#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 原因改善
        plan_mean = all_plan.mean(axis=0).astype(np.float32)
        plan_std  = all_plan.std(axis=0).astype(np.float32) + 1e-6
#        plan_mean = all_plan.mean(axis=0)
#        plan_std  = all_plan.std(axis=0) + 1e-6

    # ── 出力DTフォーマットへ詰め替え ────────────────────
    dt_trajectories = []
    for i in range(len(observations)):
        obs = observations[i]                           # (T, 19)
        act = actions[i]                                # (T, A)
        rtg = returns[i]                                # (T, 1)

        obs_norm = (obs - obs_mean) / obs_std
        rtg_norm = (rtg - ret_mean) / ret_std

        T = obs.shape[0]
        timesteps = (np.arange(T, dtype=np.int64) % TIMESTEP_MAX)

        item = {
            "observations": obs_norm.astype(np.float32),
            "actions":      act.astype(np.float32),
            "returns":      rtg_norm.astype(np.float32),
            "timesteps":    timesteps,
            "initial_rtg":  rtg[:1].copy().astype(np.float32),  # 非正規化の先頭RTG（参照用）
        }
        if have_plan:
            item["plan"] = plans[i].astype(np.float32)          # (T, 2M)

        # 最終NaNチェック
        for k in ["observations", "actions", "returns"]:
            _finite_or_raise(k, item[k])

        dt_trajectories.append(item)

    # ── 保存 ───────────────────────────────────────────
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(dt_trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)

    out_stats = {
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std":  obs_std.astype(np.float32),
        "ret_mean": ret_mean.astype(np.float32),
        "ret_std":  ret_std.astype(np.float32),
#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 原因改善
#        "count":    int(count_prev + count_new),
        "obs_version": "v2",
        "obs_dim": int(OBS_V2_DIM),
    }
    if have_plan:
        out_stats["plan_mean"] = plan_mean.astype(np.float32)
        out_stats["plan_std"]  = plan_std.astype(np.float32)

    with open(NORM_PKL, "wb") as f:
        pickle.dump(out_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ DT変換 完了: {OUTPUT_PKL}")
    print(f"   episodes={len(dt_trajectories)}, obs_dim={OBS_V2_DIM}, have_plan={have_plan}")

#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 原因改善
#    print(f"   stats -> {NORM_PKL}  (count={out_stats['count']})")

if __name__ == "__main__":
    main()
