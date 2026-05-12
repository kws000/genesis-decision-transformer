
# convert_to_dt_format.py
# ── ボトルネック認識とVmax魂の注入 5.1（snapshot管理対応：最短ルート版） ──

import os
import json
import shutil
import pickle
import datetime
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np


import json
import tempfile
import shutil

# ── パス設定 ─────────────────────────────────────────────
INPUT_PKL = "trajectories/trajectory_data.pkl"   # export_csv_to_pkl の出力

#進化ループの大改修	正規化の固定統計
BASE_NORM_PKL = "data_dt/base_mean_std.pkl"   # ★固定統計

DATA_DT_DIR = "data_dt"
SNAP_DIR = os.path.join(DATA_DT_DIR, "snapshots")
TMP_DIR = os.path.join(DATA_DT_DIR, "tmp")

DS_COLLECTION_PATH = os.path.join(DATA_DT_DIR, "ds_collection.json")

os.makedirs(DATA_DT_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ── ハイパラ ─────────────────────────────────────────────
TIMESTEP_MAX = 4096  # 位置埋め込みの語彙上限と揃える（例）

# ── 多目的RTG（終端評価→全時刻へブロードキャスト） ──
LAM_OUT = 0.35
K_TIME = 0.06

# ── OBS V2 スキーマ（参照用/検証用） ─────────────────────
OBS_V2_DIM = 19  # 既存10 + VMAX塊9


# =========================================================
# utils
# =========================================================

#進化ループの大改修	抽選改善
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


#進化ループの大改修	抽選改善
def _atomic_write_json(path: str, obj: dict):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        shutil.move(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

#進化ループの大改修	抽選改善
def _pick_bin_label(bin_rates: dict) -> str:
    # episode=1なら 1.0 のbinが必ずある想定
    best_k = None
    best_v = -1.0
    for k in ["safe", "fast", "boundary", "both"]:
        v = float(bin_rates.get(k, 0.0))
        if v > best_v:
            best_k, best_v = k, v
    return best_k or "boundary"

def _update_ds_collection_with_summary(root_dir: str, ds_id: str, rel_path: str):
    col_path  = os.path.join(root_dir, "ds_collection.json")
    meta_path = os.path.join(root_dir, rel_path, "meta.json")
    if not os.path.exists(meta_path):
        return

    meta = _load_json(meta_path)
    s = meta.get("stats", {})
    rtg = s.get("rtg_vec_initial", {})
    br  = s.get("bin_rates", {})

    # episode=1 前提：単値に落とす
    rtg_prog  = float(rtg.get("progress", {}).get("mean", 0.0))
    rtg_clean = float(rtg.get("clean", {}).get("mean", 0.0))
    bin_label = _pick_bin_label(br)

    summary = {
        "episodes": int(s.get("num_episodes", 0)),
        "rtg_prog": rtg_prog,
        "rtg_clean": rtg_clean,
        "bin": bin_label,
        # "steps": int(s.get("total_steps", 0)),  # もしmetaに入れてるなら有効化
    }

    col = _load_json(col_path)
    for it in col.get("snapshots", []):
        if it.get("ds_id") == ds_id:
            it["summary"] = summary
            break
    _atomic_write_json(col_path, col)

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _finite_or_raise(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))
        raise RuntimeError(f"[convert_to_dt_format] {name} contains NaN/Inf at indices {bad}")


def _now_iso_jst() -> str:
    # ユーザー環境は JST 前提でOK（ログ/管理が楽）
    jst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(jst).isoformat(timespec="seconds")


def _try_get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return r.stdout.strip()[:12]
    except Exception:
        return "UNKNOWN"


def _atomic_write_json(path: str, obj: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_or_init_ds_collection() -> Dict[str, Any]:
    if os.path.exists(DS_COLLECTION_PATH):
        with open(DS_COLLECTION_PATH, "r", encoding="utf-8") as f:
            col = json.load(f)
        # 最低限の保険
        if "version" not in col:
            col["version"] = 1
        if "next_id" not in col:
            # 既存snapshotsから推測（あれば）
            col["next_id"] = 1
        if "snapshots" not in col:
            col["snapshots"] = []
        return col

    # 初期化
    return {
        "version": 1,
        "next_id": 1,
        "snapshots": []
    }


def _allocate_ds_id_and_update_collection(
    num_episodes: int,
    context_len_hint: int,
) -> Dict[str, Any]:
    """
    ds_collection.json の next_id を消費して採番し、snapshotsへレコード追加して保存。
    戻り値: {"ds_id":..., "rel_path":..., "abs_path":...}
    """
    col = _load_or_init_ds_collection()

    next_id = int(col.get("next_id", 1))
    ds_id = f"ds_{next_id:06d}"
    rel_path = f"snapshots/{ds_id}"
    abs_path = os.path.join(DATA_DT_DIR, rel_path)

    record = {
        "ds_id": ds_id,
        "path": rel_path,
        "created_at": _now_iso_jst(),
        "num_episodes": int(num_episodes),
        "context_len_hint": int(context_len_hint),
    }

    # 追記
    col["snapshots"].append(record)
    col["next_id"] = next_id + 1

    # 保存（atomic）
    _atomic_write_json(DS_COLLECTION_PATH, col)

    return {"ds_id": ds_id, "rel_path": rel_path, "abs_path": abs_path}


def _write_meta_json(
    snapshot_abs_dir: str,
    ds_id: str,
    num_episodes: int,
    obs_dim: int,
    action_dim: int,
    returns_vec_dim: int,
    have_plan: bool,
    notes: str = "",
    dt_trajectories=None,              # ★追加
    prog_fast_thr: float = 0.75,       # ★追加（今は固定でOK）
    clean_safe_thr: float = 0.75,      # ★追加    
):
    meta = {
        "version": 1,
        "ds_id": ds_id,
        "created_at": _now_iso_jst(),
        "source": {
            "generator": "convert_to_dt_format.py",
            "git_commit": _try_get_git_commit(),
            "input_pkl": INPUT_PKL,
        },
        "stats": {
            "num_episodes": int(num_episodes),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "returns_vec_dim": int(returns_vec_dim),
            "have_plan": bool(have_plan),
        },
        "notes": notes,
    }

    # meta に追加情報 rtg_vec / bin の可視化統計（エピソード単位） ──
    if dt_trajectories is not None and len(dt_trajectories) > 0:
        # initial_rtg: (1,2) を (E,2) に集計
        init = np.stack([tr["initial_rtg"][0] for tr in dt_trajectories], axis=0).astype(np.float32)  # (E,2)
        prog = init[:, 0]
        clean = init[:, 1]

        def _stats(v: np.ndarray):
            v = v.astype(np.float32)
            return {
                "mean": float(v.mean()),
                "std":  float(v.std() + 1e-9),
                "min":  float(v.min()),
                "max":  float(v.max()),
                "p10":  float(np.percentile(v, 10)),
                "p50":  float(np.percentile(v, 50)),
                "p90":  float(np.percentile(v, 90)),
            }

        # bin分類（エピソード単位）
        bins = {"safe": 0, "fast": 0, "boundary": 0, "both": 0}
        for p, c in zip(prog, clean):
            is_fast = (p >= prog_fast_thr)
            is_safe = (c >= clean_safe_thr)
            if is_fast and is_safe:
                bins["both"] += 1
            elif is_safe and (not is_fast):
                bins["safe"] += 1
            elif (not is_safe) and is_fast:
                bins["fast"] += 1
            else:
                bins["boundary"] += 1

        E = int(len(dt_trajectories))
        rates = {k: (v / max(1, E)) for k, v in bins.items()}

        # metaに追加
        meta["stats"]["rtg_vec_initial"] = {
            "progress": _stats(prog),
            "clean": _stats(clean),
        }
        meta["stats"]["binning"] = {
            "prog_fast_thr": float(prog_fast_thr),
            "clean_safe_thr": float(clean_safe_thr),
        }
        meta["stats"]["bin_counts"] = {k: int(v) for k, v in bins.items()}
        meta["stats"]["bin_rates"]  = {k: float(v) for k, v in rates.items()}

    meta_path = os.path.join(snapshot_abs_dir, "meta.json")
    _atomic_write_json(meta_path, meta)


def _finalize_snapshot(tmp_dt_pkl: str, tmp_norm_pkl: str, ds_id: str, snapshot_abs_dir: str):
    os.makedirs(snapshot_abs_dir, exist_ok=False)  # 事故防止：既存なら落とす
    shutil.move(tmp_dt_pkl, os.path.join(snapshot_abs_dir, "trajectories_dt.pkl"))

#進化ループの大改修	正規化の固定統計 一度作ったら二度と更新してはいけない
#    shutil.move(tmp_norm_pkl, os.path.join(snapshot_abs_dir, "mean_std.pkl"))


# =========================================================
# main
# =========================================================

def main():
    # ── 入力ロード ───────────────────────────────────────
    with open(INPUT_PKL, "rb") as f:
        raw = pickle.load(f)

    # 形式: [trajectory, ...] 想定（collect_trajectory）
    if isinstance(raw, dict):
        raw = [raw]

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    returns_vec: List[np.ndarray] = []
    plans: List[np.ndarray] = []

    #進化ループの大改修	計画不具合改善
    wp_preview_list = []   # ★ ループ前に追加しておく（observations, actions...と並べる）

    have_plan = False

    progress_score = 0.0
    clean_score    = 0.0

    for traj in raw:
        obs = _ensure_2d(np.asarray(traj["obs"], dtype=np.float32))          # (T, 19)
        act = _ensure_2d(np.asarray(traj["action"], dtype=np.float32))       # (T, A)
        rew = np.asarray(traj["reward"], dtype=np.float32).reshape(-1)       # (T,)
        pln = traj.get("plan", None)                                         # (T, 2M) or None

        #進化ループの大改修	計画不具合改善
        if pln is not None:
            pln = _ensure_2d(np.asarray(pln, dtype=np.float32))  # (T, 6) 想定
            assert pln.shape[0] == obs.shape[0], "T mismatch in plan"

			#進化ループの大改修	計画不具合改善 ここで「階段状(plan hold)」にする
            H = 5  # 例：5フレームごとにしか更新しない（後でds_blenderへ移してもOK）
            idx = (np.arange(pln.shape[0]) // H) * H   # 0,0,0,0,0,5,5,5,5,5,10,10,...
            pln = pln[idx]                              # (T,6) のまま、内容だけ階段状

            plans.append(pln)
            have_plan = True

            # ★ wp_preview を「前回 plan」として作る（T, K=3, 5）
            T = pln.shape[0]
            wp_prev = np.zeros((T, 3, 5), dtype=np.float32)
            # t=0 はゼロ（前回が無いので）
            wp_prev[1:, 0, 0:2] = pln[:-1, 0:2]  # (x1,y1)
            wp_prev[1:, 1, 0:2] = pln[:-1, 2:4]  # (x2,y2)
            wp_prev[1:, 2, 0:2] = pln[:-1, 4:6]  # (x3,y3)
            wp_preview_list.append(wp_prev)
        else:
            # planが無いエピソードはwpも無し（後段でゼロ扱い）
            wp_preview_list.append(None)
            
        # 形の基本検証
        assert obs.shape[1] == OBS_V2_DIM, f"obs_dim {obs.shape[1]} != {OBS_V2_DIM}"
        assert obs.shape[0] == act.shape[0] == rew.shape[0], "T mismatch among obs/act/reward"

        # ── 多目的RTG（Progress, Clean） ───────────────────────
        ep_time = float(traj["ep_time"])
        n_out   = float(traj["n_out"])
        progress_score = float(np.exp(-K_TIME * ep_time))
        clean_score    = float(np.exp(-LAM_OUT * n_out))
        rtg_vec = np.tile(
            np.array([progress_score, clean_score], dtype=np.float32),
            (obs.shape[0], 1)
        )  # (T,2)

#※二回appendしてしまっている！
#        if pln is not None:
#            pln = _ensure_2d(np.asarray(pln, dtype=np.float32))  # (T, 2M)
#            assert pln.shape[0] == obs.shape[0], "T mismatch in plan"
#            plans.append(pln)
#            have_plan = True

        observations.append(obs)
        actions.append(act)
        returns_vec.append(rtg_vec)

    # ── 正規化統計 ──────────────────────────────────────
    all_obs = np.concatenate(observations, axis=0).astype(np.float32)      # (∑T, 19)
    all_rtg_vec = np.concatenate(returns_vec, axis=0).astype(np.float32)   # (∑T, 2)
    all_plan = np.concatenate(plans, axis=0).astype(np.float32) if have_plan else None

    _finite_or_raise("observations", all_obs)
    _finite_or_raise("returns_vec", all_rtg_vec)
    if have_plan:
        _finite_or_raise("plan", all_plan)

    # std 下限での止血（既定値はあなたの現状を踏襲）
    obs_mean = all_obs.mean(axis=0).astype(np.float32)
    obs_std  = all_obs.std(axis=0).astype(np.float32)
    obs_std  = np.maximum(obs_std, 0.05).astype(np.float32)

	#進化ループの大改修	正規化の固定統計
    if os.path.exists(BASE_NORM_PKL):
        # ※一度作ったら二度と更新してはいけない厳しい、、
        with open(BASE_NORM_PKL, "rb") as f:
            base = pickle.load(f)
        base_mean = np.asarray(base["obs_mean"], dtype=np.float32)
        base_std  = np.asarray(base["obs_std"],  dtype=np.float32)
        assert base_mean.shape[0] == OBS_V2_DIM, f"base obs_dim mismatch: {base_mean.shape}"
        assert base_std.shape[0]  == OBS_V2_DIM, f"base obs_dim mismatch: {base_std.shape}"
        print(f"✅ base norm loaded: {BASE_NORM_PKL}")
    else:
        base_mean = obs_mean.copy()
        base_std  = obs_std.copy()
        with open(BASE_NORM_PKL, "wb") as f:
            pickle.dump({
                "obs_mean": base_mean.astype(np.float32),
                "obs_std":  base_std.astype(np.float32),
                "obs_version": "v2",
                "obs_dim": int(OBS_V2_DIM),
                "note": "fixed/global normalization for mixing snapshots"
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ base norm created: {BASE_NORM_PKL}")

    # このconvertで実際に使う統計は base に固定
    obs_mean = base_mean
    obs_std  = base_std

    # returns_vec の統計は保存だけ（あなたの方針：z-scoreしない）
    retv_mean = all_rtg_vec.mean(axis=0).astype(np.float32)          # (2,)
    retv_std  = all_rtg_vec.std(axis=0).astype(np.float32) + 1e-6    # (2,)

    if have_plan:
        plan_mean = all_plan.mean(axis=0).astype(np.float32)
        plan_std  = all_plan.std(axis=0).astype(np.float32) + 1e-6

    # ── 出力DTフォーマットへ詰め替え ────────────────────
    dt_trajectories: List[Dict[str, Any]] = []
    for i in range(len(observations)):
        obs = observations[i]  # (T, 19)
        act = actions[i]       # (T, A)
        rtg_v = returns_vec[i] # (T,2) 非正規化

        obs_norm = (obs - obs_mean) / obs_std
        rtg_v_norm = rtg_v  # 方針：そのまま

        T = obs.shape[0]
        timesteps = (np.arange(T, dtype=np.int64) % TIMESTEP_MAX)

        item = {
            "observations": obs_norm.astype(np.float32),
            "actions":      act.astype(np.float32),
            "returns":      rtg_v_norm.astype(np.float32),  # (T,2)
            "timesteps":    timesteps,
            "initial_rtg":  rtg_v_norm[:1].copy().astype(np.float32),  # (1,2)
        }

        if have_plan:
            item["plan"] = plans[i].astype(np.float32)

        #進化ループの大改修	計画不具合改善
        wp_prev = wp_preview_list[i]
        if wp_prev is not None:
            item["wp_preview"] = wp_prev.astype(np.float32)   # (T,3,5)

        for k in ["observations", "actions", "returns"]:
            _finite_or_raise(k, item[k])


        dt_trajectories.append(item)

    # ── tmp 保存（→ snapshotへ凍結）──────────────────────
    tmp_dt_pkl = os.path.join(TMP_DIR, "trajectories_dt.tmp.pkl")
    tmp_norm_pkl = os.path.join(TMP_DIR, "mean_std.tmp.pkl")

    with open(tmp_dt_pkl, "wb") as f:
        pickle.dump(dt_trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)

    out_stats: Dict[str, Any] = {
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std":  obs_std.astype(np.float32),
        "ret_mean": retv_mean.astype(np.float32),
        "ret_std":  retv_std.astype(np.float32),
        "ret_dim":  2,
        "obs_version": "v2",
        "obs_dim": int(OBS_V2_DIM),
    }
    if have_plan:
        out_stats["plan_mean"] = plan_mean.astype(np.float32)
        out_stats["plan_std"]  = plan_std.astype(np.float32)

    with open(tmp_norm_pkl, "wb") as f:
        pickle.dump(out_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── snapshot採番 & ds_collection更新 ───────────────────
    action_dim = int(actions[0].shape[1]) if len(actions) > 0 else 0
    info = _allocate_ds_id_and_update_collection(
        num_episodes=len(dt_trajectories),
        context_len_hint=0,  # ここは train 側で決めるので、ヒント不要なら 0 でOK
    )
    ds_id = info["ds_id"]
    snapshot_abs_dir = info["abs_path"]

    # ── snapshot凍結（move） ─────────────────────────────
    _finalize_snapshot(tmp_dt_pkl, tmp_norm_pkl, ds_id, snapshot_abs_dir)

    # ── meta.json ───────────────────────────────────────
    _write_meta_json(
        snapshot_abs_dir=snapshot_abs_dir,
        ds_id=ds_id,
        num_episodes=len(dt_trajectories),
        obs_dim=OBS_V2_DIM,
        action_dim=action_dim,
        returns_vec_dim=2,
        have_plan=have_plan,
        notes="auto snapshot from convert_to_dt_format.py (vmax integrated)",
        dt_trajectories=dt_trajectories,              # ★追加
    )

    #進化ループの大改修	抽選改善
    _update_ds_collection_with_summary("data_dt", ds_id, f"snapshots/{ds_id}")

 
    print(f"✅ DT変換 & Snapshot 凍結 完了: {ds_id}")
    print(f"   path={os.path.join('data_dt', info['rel_path'])}")
    print(f"   episodes={len(dt_trajectories)}, obs_dim={OBS_V2_DIM}, act_dim={action_dim}, have_plan={have_plan}")


if __name__ == "__main__":
    main()