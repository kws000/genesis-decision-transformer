# vehicle_control_dt/train_dt.py

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt

from model_dt import DecisionTransformer


# --- 設定 ---

TIMESTEP_MAX = 4000

## DTのMLP化検証
#context_len = 1

## DTのMLP化検証 復元step1
#context_len = 1

## DTのMLP化検証 復元step2
#context_len = 5

## DTのMLP化検証 復元step3
#context_len = 5

## DTのMLP化検証 復元step4
#context_len = 3

## DTのMLP化検証 復元step5
#context_len = 3

## DTのMLP化検証 復元step6
#context_len = 3

## DTのMLP化検証 復元step7
#context_len = 20#3

# DTのMLP化検証 復元step8
context_len = 1#5

## DTのMLP化検証 復元
#context_len = 20

embed_dim = 128
# === ハイパーパラメータ ===

#計画と行動のマルチタスクモデル
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
K_WP = 40
PLAN_M = 3
W_ACT = 1.0
W_PLAN = 0.5
W_SMOOTH = 0.01
USE_FOCUS = False
#BATCH_SIZE = 32
#EPOCHS = 100
#LR = 1e-3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pkl_path = "data_dt/trajectories_dt.pkl"
model_path = "models/decision_transformer.pt"

#進化ループの大改修	正規化の固定統計
BASE_NORM_PKL = "data_dt/base_mean_std.pkl"   # ★固定統計

# --- Dataset定義 ---
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

import os, json, pickle
import numpy as np
import torch
from torch.utils.data import Dataset

# =========================================================
# Snapshot + Bin Mixer Dataset（互換不要 / 最短ルート）
# =========================================================
#進化ループの大改修 サンプラー対応（SequenceDataset廃止）
class SnapshotBinMixerDataset(Dataset):
    def __init__(self, root_dir, context_len, num_samples=128_000,
                 ds_collection_name="ds_collection.json",
                 ds_blender_name="ds_blender.json",
                 seed=0):
        self.root_dir = root_dir
        self.context_len = int(context_len)
        self.num_samples = int(num_samples)

        self.collection_path = os.path.join(root_dir, ds_collection_name)
        self.blender_path = os.path.join(root_dir, ds_blender_name)

        with open(self.collection_path, "r", encoding="utf-8") as f:
            self.collection = json.load(f)
        with open(self.blender_path, "r", encoding="utf-8") as f:
            self.blender = json.load(f)

        # --------------------------
        # (A) bins / thresholds
        # --------------------------
        binning = self.blender.get("binning", {})
        self.prog_fast_thr = float(binning.get("prog_fast_thr", 0.75))
        self.clean_safe_thr = float(binning.get("clean_safe_thr", 0.75))

        self.bin_names = binning.get("bins", ["safe", "fast", "boundary", "both"])
        assert set(self.bin_names) == {"safe", "fast", "boundary", "both"}

        bin_mix = self.blender.get("bin_mix", {})
        bw = np.array([float(bin_mix.get(b, 0.0)) for b in self.bin_names], dtype=np.float64)
        bw = np.maximum(bw, 0.0)
        bw = bw / max(1e-12, bw.sum())
        self.bin_weights = bw.astype(np.float64)

        self.rng = np.random.default_rng(seed)

        # --------------------------
        # (B) episode_weight settings
        # --------------------------
        ew = self.blender.get("episode_weight", {}) or {}
        self.ew_mode  = str(ew.get("mode", "linear"))
        self.ew_floor = float(ew.get("floor", 0.2))      # 外れも引くための下駄
        self.ew_cap   = float(ew.get("cap", 2.0))        # 上限
        self.ew_a     = float(ew.get("a_prog", 0.5))
        self.ew_b     = float(ew.get("b_clean", 0.5))

        # --------------------------
        # (C) candidate episodes = snapshots
        # snapshot_mix があれば「候補集合＋事前重み」として利用
        # --------------------------
        snap_mix = self.blender.get("snapshot_mix", [])
        snap_prior = {}
        if isinstance(snap_mix, list) and len(snap_mix) > 0:
            for x in snap_mix:
                did = x.get("ds_id")
                if did:
                    snap_prior[did] = float(x.get("weight", 1.0))
        # priorが空なら、collectionの全dsを候補にする
        self.candidate_ds = set(snap_prior.keys()) if len(snap_prior) > 0 else None

        # ds_id -> rel snapshot dir
        snap_map = {}
        for s in self.collection.get("snapshots", []):
            if "ds_id" not in s:
                continue
            ds_id = s["ds_id"]
            if (self.candidate_ds is not None) and (ds_id not in self.candidate_ds):
                continue
            snap_map[ds_id] = s.get("path", f"snapshots/{ds_id}")
        if len(snap_map) == 0:
            raise RuntimeError("No snapshots found in ds_collection for current candidate set.")

        self.ds_ids = list(snap_map.keys())
        self.ds_relpath = {ds_id: snap_map[ds_id] for ds_id in self.ds_ids}

        # --------------------------
        # (D) build bin -> episodes (ds_id) with weights
        # --------------------------
        def _get_prog_clean_from_summary(summ: dict):
            # 新形式: rtg_prog/rtg_clean
            if "rtg_prog" in summ and "rtg_clean" in summ:
                return float(summ.get("rtg_prog", 0.0)), float(summ.get("rtg_clean", 0.0))
            # 旧形式: prog_mean/clean_mean
            if "prog_mean" in summ and "clean_mean" in summ:
                return float(summ.get("prog_mean", 0.0)), float(summ.get("clean_mean", 0.0))
            # 最低限
            return 0.0, 0.0

        def _get_bin_label_from_summary(summ: dict, prog: float, clean: float):
            # 新形式: summ["bin"]
            b = summ.get("bin", None)
            if isinstance(b, str) and b in self.bin_names:
                return b
            # 旧形式: bin_rates
            br = summ.get("bin_rates", None)
            if isinstance(br, dict):
                best_k, best_v = None, -1.0
                for k in self.bin_names:
                    v = float(br.get(k, 0.0))
                    if v > best_v:
                        best_k, best_v = k, v
                if best_k is not None:
                    return best_k
            # fallback: thresholdsで分類
            return self._classify_bin(prog, clean)

        def _episode_weight(prog: float, clean: float):
            # v1: linear + floor + cap
            bonus = self.ew_a * prog + self.ew_b * clean
            # bonusは負になり得るので0下限
            bonus = max(0.0, bonus)
            w = self.ew_floor + min(self.ew_cap - self.ew_floor, bonus)
            return float(max(1e-9, w))

        self.bin_to_eps = {b: [] for b in self.bin_names}   # bin -> [(ds_id, w), ...]
        for rec in self.collection.get("snapshots", []):
            ds_id = rec.get("ds_id", None)
            if ds_id is None:
                continue
            if ds_id not in self.ds_relpath:
                continue

            summ = rec.get("summary", {}) or {}
            prog, clean = _get_prog_clean_from_summary(summ)
            b = _get_bin_label_from_summary(summ, prog, clean)

            wq = _episode_weight(prog, clean)
            wp = float(snap_prior.get(ds_id, 1.0))  # snapshot_mix weight (なければ1)
            w = max(1e-9, wq * max(0.0, wp))

            self.bin_to_eps[b].append((ds_id, w))

        # binが空の場合は後でfallbackするが、全部空はNG
        total_eps = sum(len(v) for v in self.bin_to_eps.values())
        if total_eps == 0:
            raise RuntimeError("No episodes registered into bin_to_eps. Check ds_collection.summary.")

        # --------------------------
        # (E) load trajectories + precompute per-ds per-bin indices (as before)
        # --------------------------
        self.ds_data = {}
        self.ds_bins = {}
        self.k_wp = None
        self.plan_dim = 2 * PLAN_M

        # ---- stats (single-process only: use num_workers=0) ----
        self._stats = {
            "total": 0,
            "by_ds": {ds_id: 0 for ds_id in self.ds_ids},
            "by_bin": {b: 0 for b in self.bin_names},
            "by_ds_bin": {ds_id: {b: 0 for b in self.bin_names} for ds_id in self.ds_ids},
        }

        for ds_id in self.ds_ids:
            snap_dir = os.path.join(root_dir, self.ds_relpath[ds_id])
            traj_path = os.path.join(snap_dir, "trajectories_dt.pkl")
            if not os.path.exists(traj_path):
                raise FileNotFoundError(f"missing trajectories_dt.pkl: {traj_path}")

            with open(traj_path, "rb") as f:
                trajs = pickle.load(f)

            bins = {b: [] for b in self.bin_names}

            for j, tr in enumerate(trajs):
                obs = tr["observations"]
                T = int(len(obs))
                if T < self.context_len:
                    continue

                init = tr.get("initial_rtg", None)
                if init is None:
                    continue
                init = np.asarray(init, dtype=np.float32).reshape(-1)
                if init.shape[0] < 2:
                    continue

                prog = float(init[0])
                clean = float(init[1])
                b = self._classify_bin(prog, clean)
                bins[b].append(j)

                if self.k_wp is None:
                    wpv = tr.get("wp_preview", None)
                    if isinstance(wpv, np.ndarray):
                        if wpv.ndim == 3:
                            self.k_wp = int(wpv.shape[1])
                        elif wpv.ndim == 2:
                            self.k_wp = int(wpv.shape[0])

            if self.k_wp is None:
                self.k_wp = K_WP

            self.ds_data[ds_id] = trajs
            self.ds_bins[ds_id] = bins

        # sanity
        total_usable = 0
        for ds_id in self.ds_ids:
            total_usable += sum(len(self.ds_bins[ds_id][b]) for b in self.bin_names)
        if total_usable == 0:
            raise RuntimeError("No usable trajectories (check initial_rtg exists and T>=context_len).")

        # 学習効率が落ちている原因探し auto scale num_samples (optional) 
        coverage = float(self.blender.get("sampling", {}).get("epoch_coverage", 0.0))
        max_ns   = int(self.blender.get("sampling", {}).get("max_num_samples", 1_000_000))

        if coverage > 0.0:
            # 候補ds（ロード済み）について window総数を概算
            total_windows = 0
            for ds_id in self.ds_ids:
                for b in self.bin_names:
                    # bins[b] は trajectory index のリスト。今はds=1epが多いので len(bins[b]) はほぼ0/1
                    for j in self.ds_bins[ds_id][b]:
                        T = len(self.ds_data[ds_id][j]["observations"])
                        total_windows += max(0, T - self.context_len)
            target = int(min(max_ns, max(1, coverage * total_windows)))
            self.num_samples = target
            print(f"[SnapshotBinMixerDataset] auto num_samples={self.num_samples} (coverage={coverage}, total_windows={total_windows})")




    def get_norm_path_for_training(self) -> str:
        # 最短：固定統計（BASE_NORM_PKL）を使う
        if not os.path.exists(BASE_NORM_PKL):
            raise FileNotFoundError(f"missing mean_std.pkl: {BASE_NORM_PKL}")
        return BASE_NORM_PKL

    def get_and_reset_stats(self):
        s = self._stats
        out = {
            "total": int(s["total"]),
            "by_ds": {k: int(v) for k, v in s["by_ds"].items()},
            "by_bin": {k: int(v) for k, v in s["by_bin"].items()},
            "by_ds_bin": {ds: {b: int(v) for b, v in bd.items()} for ds, bd in s["by_ds_bin"].items()},
        }
        self._stats["total"] = 0
        for ds in self._stats["by_ds"]:
            self._stats["by_ds"][ds] = 0
            for b in self._stats["by_ds_bin"][ds]:
                self._stats["by_ds_bin"][ds][b] = 0
        for b in self._stats["by_bin"]:
            self._stats["by_bin"][b] = 0
        return out

    def _classify_bin(self, prog: float, clean: float) -> str:
        both = (prog >= self.prog_fast_thr) and (clean >= self.clean_safe_thr)
        if both:
            return "both"
        safe = (clean >= self.clean_safe_thr) and (prog < self.prog_fast_thr)
        if safe:
            return "safe"
        fast = (prog >= self.prog_fast_thr) and (clean < self.clean_safe_thr)
        if fast:
            return "fast"
        return "boundary"

    def __len__(self):
        return self.num_samples

    def _pick_bin(self) -> str:
        idx = self.rng.choice(len(self.bin_names), p=self.bin_weights)
        return self.bin_names[int(idx)]

    def _pick_ds_from_bin(self, bin_name: str) -> str:
        eps = self.bin_to_eps.get(bin_name, [])
        if len(eps) == 0:
            # fallback: どこかのbinから選ぶ
            all_eps = []
            for b in self.bin_names:
                all_eps.extend(self.bin_to_eps.get(b, []))
            if len(all_eps) == 0:
                raise RuntimeError("No episodes available in bin_to_eps.")
            eps = all_eps

        ds_list = [d for d, _ in eps]
        w = np.array([float(w) for _, w in eps], dtype=np.float64)
        w = np.maximum(w, 0.0)
        w = w / max(1e-12, w.sum())
        idx = self.rng.choice(len(ds_list), p=w)
        return ds_list[int(idx)]

    def _fallback_pick_traj_index(self, ds_id: str) -> int:
        all_list = []
        for b in self.bin_names:
            all_list.extend(self.ds_bins[ds_id][b])
        if len(all_list) == 0:
            # change ds by scanning bins
            for _ in range(10):
                b2 = self._pick_bin()
                ds2 = self._pick_ds_from_bin(b2)
                all2 = []
                for b in self.bin_names:
                    all2.extend(self.ds_bins[ds2][b])
                if len(all2) > 0:
                    return int(self.rng.choice(all2))
            raise RuntimeError("All bins empty across episodes. Check dataset generation.")
        return int(self.rng.choice(all_list))

    def __getitem__(self, idx):
        # 1) bin を先に決める（学習分布の骨格）
        bin_name = self._pick_bin()

        # 2) その bin に属する episode(ds) を「優秀ほど出やすいが外れもある」重みで抽選
        ds_id = self._pick_ds_from_bin(bin_name)

        # 3) ds内で該当binの trajectory index を選ぶ（なければfallback）
        candidates = self.ds_bins[ds_id][bin_name]
        if len(candidates) == 0:
            ok = False
            for _ in range(4):
                b2 = self._pick_bin()
                c2 = self.ds_bins[ds_id][b2]
                if len(c2) > 0:
                    bin_name = b2
                    candidates = c2
                    ok = True
                    break
            if not ok:
                traj_idx = self._fallback_pick_traj_index(ds_id)
            else:
                traj_idx = int(self.rng.choice(candidates))
        else:
            traj_idx = int(self.rng.choice(candidates))

        # ---- stats ----
        self._stats["total"] += 1
        self._stats["by_ds"][ds_id] += 1
        self._stats["by_bin"][bin_name] += 1
        self._stats["by_ds_bin"][ds_id][bin_name] += 1

        tr = self.ds_data[ds_id][traj_idx]

        obs = tr["observations"]
        act = tr["actions"]
        ret = tr["returns"]
        tms = tr["timesteps"]

        T = int(len(obs))
        max_start = T - self.context_len
        start = 0 if max_start <= 0 else int(self.rng.integers(0, max_start + 1))

        obs_seq = obs[start:start + self.context_len]
        act_seq = act[start:start + self.context_len]
        rtg_seq = ret[start:start + self.context_len]      # (L,2)
        tms_seq = tms[start:start + self.context_len]

        wpv = tr.get("wp_preview", None)
        plan = tr.get("plan", None)

        # wp
        if isinstance(wpv, np.ndarray):
            if wpv.ndim == 3:
                wp_k5 = wpv[start + self.context_len - 1]
            elif wpv.ndim == 2:
                wp_k5 = wpv
            else:
                wp_k5 = np.zeros((self.k_wp, 5), dtype=np.float32)
        else:
            wp_k5 = np.zeros((self.k_wp, 5), dtype=np.float32)

        if wp_k5.shape[0] < self.k_wp:
            pad = np.zeros((self.k_wp - wp_k5.shape[0], 5), dtype=wp_k5.dtype)
            wp_k5 = np.concatenate([wp_k5, pad], axis=0)
        elif wp_k5.shape[0] > self.k_wp:
            wp_k5 = wp_k5[:self.k_wp]

        # plan
        if isinstance(plan, np.ndarray):
            if plan.ndim == 2:
                plan_vec = plan[start + self.context_len - 1]
            else:
                plan_vec = plan
            plan_vec = np.asarray(plan_vec, dtype=np.float32).reshape(-1)
            if plan_vec.shape[0] < self.plan_dim:
                pad = np.zeros((self.plan_dim - plan_vec.shape[0],), dtype=np.float32)
                plan_vec = np.concatenate([plan_vec, pad], axis=0)
            elif plan_vec.shape[0] > self.plan_dim:
                plan_vec = plan_vec[:self.plan_dim]
        else:
            plan_vec = np.zeros((self.plan_dim,), dtype=np.float32)

        return (
            torch.tensor(obs_seq, dtype=torch.float32),
            torch.tensor(act_seq, dtype=torch.float32),
            torch.tensor(rtg_seq, dtype=torch.float32),
            torch.tensor(tms_seq, dtype=torch.long),
            torch.tensor(wp_k5, dtype=torch.float32),
            torch.tensor(plan_vec, dtype=torch.float32),
        )
            
# 学習サンプリング方式の変更 ※廃止
class SequenceDataset(Dataset):

#計画と行動のマルチタスクモデル    
    def __init__(self, path, context_len):
        # 引数名の取り違え修正（pkl_path -> path）
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
#   def __init__(self, path, context_len):
#       with open(pkl_path, "rb") as f:
#            trajectories = pickle.load(f)

        self.context_len = context_len
        self.samples = []

#計画と行動のマルチタスクモデル    
        # データセット全体で固定のKを決める（バッチ結合のため）
        target_K = None
        for tr in trajectories:
            wpv = tr.get("wp_preview", None)
            if isinstance(wpv, np.ndarray):
                if wpv.ndim == 3:      # (T,K,5)
                    target_K = int(wpv.shape[1]); break
                elif wpv.ndim == 2:    # (K,5)
                    target_K = int(wpv.shape[0]); break
        if target_K is None:
            target_K = K_WP
        self.k_wp = target_K
        self.plan_dim = 2 * PLAN_M

        for traj in trajectories:
            obs = traj["observations"]
            act = traj["actions"]
            ret = traj["returns"]
            tms = traj["timesteps"]

#計画と行動のマルチタスクモデル 追加フィールド（あれば利用）
            wpv = traj.get("wp_preview", None)   # (T,K,5) or (K,5)
            plan = traj.get("plan", None)        # (2M,) or (T,2M)
#           initial_rtg = traj["initial_rtg"][0]


            T = len(obs)
            if T < context_len:
                continue

            for i in range(T - context_len):

                # 初期報酬を渡すように
                obs_seq = obs[i:i+context_len]
                act_seq = act[i:i+context_len]
                rtg_seq = ret[i:i+context_len]
                tms_seq = tms[i:i+context_len]

#計画と行動のマルチタスクモデル
                # --- WPプレフィクス（窓の末尾に合わせる） ---
                if isinstance(wpv, np.ndarray):
                    if wpv.ndim == 3:      # (T,K,5)
                        wp_k5 = wpv[i + context_len - 1]
                    elif wpv.ndim == 2:    # (K,5)
                        wp_k5 = wpv
                    else:
                        wp_k5 = np.zeros((self.k_wp, 5), dtype=np.float32)
                else:
                    wp_k5 = np.zeros((self.k_wp, 5), dtype=np.float32)
                # pad/trim to K
                if wp_k5.shape[0] < self.k_wp:
                    pad = np.zeros((self.k_wp - wp_k5.shape[0], 5), dtype=wp_k5.dtype)
                    wp_k5 = np.concatenate([wp_k5, pad], axis=0)
                elif wp_k5.shape[0] > self.k_wp:
                    wp_k5 = wp_k5[:self.k_wp]
                # --- 計画ラベル（なければゼロで占位） ---
                if isinstance(plan, np.ndarray):
                    if plan.ndim == 2:      # (T, 2M)
                        plan_vec = plan[i + context_len - 1]
                    else:                    # (2M,)
                        plan_vec = plan
                    # pad/trim to 2*PLAN_M
                    if plan_vec.shape[0] < self.plan_dim:
                        pad = np.zeros((self.plan_dim - plan_vec.shape[0],), dtype=np.float32)
                        plan_vec = np.concatenate([plan_vec.astype(np.float32), pad], axis=0)
                    elif plan_vec.shape[0] > self.plan_dim:
                        plan_vec = plan_vec[:self.plan_dim].astype(np.float32)
                else:
                    plan_vec = np.zeros((self.plan_dim,), dtype=np.float32)
                self.samples.append({
                    "states":    obs_seq,
                    "actions":   act_seq,
                    "returns":   rtg_seq,
                    "timesteps": tms_seq,
                    "wp":        wp_k5,      # (K,5)
                    "plan":      plan_vec,   # (2*PLAN_M,)
                })
#               self.samples.append({
#                   "states": obs_seq,
#                   "actions": act_seq,
#                   "returns": rtg_seq,
#                   "timesteps": tms_seq,
#               })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

# 未来報酬の特徴分離	RTGベクトルを追加
        rets = sample["returns"]  # (T,2) だけ
        return (
            torch.tensor(sample["states"],    dtype=torch.float32),
            torch.tensor(sample["actions"],   dtype=torch.float32),
            torch.tensor(rets,                dtype=torch.float32),  # (T,2)
            torch.tensor(sample["timesteps"], dtype=torch.long),
            torch.tensor(sample["wp"],        dtype=torch.float32),
            torch.tensor(sample["plan"],      dtype=torch.float32),
        )
#        #計画と行動のマルチタスクモデル
#        rets = sample["returns"]
#        if rets.ndim == 1:
#            rets = rets[:, None]  # (T,) -> (T,1)
#        return (
#            torch.tensor(sample["states"],    dtype=torch.float32),  # (T, obs_dim)
#            torch.tensor(sample["actions"],   dtype=torch.float32),  # (T, act_dim)
#            torch.tensor(rets,                dtype=torch.float32),  # (T, 1)
#            torch.tensor(sample["timesteps"], dtype=torch.long),     # (T,)
#            torch.tensor(sample["wp"],        dtype=torch.float32),  # (K, 5)
#            torch.tensor(sample["plan"],      dtype=torch.float32),  # (2*PLAN_M,)
#        )



class TrajectoryDataset(Dataset):
    def __init__(self, path, context_len):
        with open(path, "rb") as f:
            trajectories = pickle.load(f)

        # 偏り可視化
        debug_actions = np.concatenate([traj["actions"] for traj in trajectories], axis=0)
        print("steering mean:", debug_actions[:, 0].mean())
        print("steering std :", debug_actions[:, 0].std())

        # 全アクションをまとめてプロット
        is_plot = False#True#False
        if is_plot:

            all_steerings = []

            for traj in trajectories:
                actions = traj["actions"]  # shape: (T, 2)
                steer = actions[:, 0]      # steering 成分だけ取り出す
                all_steerings.extend(steer.tolist())

            plt.hist(all_steerings, bins=50, alpha=0.7)
            plt.title("Steering Distribution")
            plt.xlabel("Steering Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

        self.obs = []
        self.actions = []
        self.returns = []
        self.timesteps = []

        for traj in trajectories:
            self.obs.append(traj["observations"])
            self.actions.append(traj["actions"])
            self.returns.append(traj["returns"])
            self.timesteps.append(traj["timesteps"])

        self.obs = np.array(self.obs)
        self.actions = np.array(self.actions)
        self.returns = np.array(self.returns)
        self.timesteps = np.array(self.timesteps)

        self.context_len = context_len
        self.length = len(self.obs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        o = self.obs[idx]
        a = self.actions[idx]
        r = self.returns[idx]
        t = self.timesteps[idx]
        # 👈 IndexError を防ぐ！
        t = t % TIMESTEP_MAX

        # (T,) → (T, 1) に変換（次元揃え）
        if o.ndim == 1:
            o = o[:, None]
        if a.ndim == 1:
            a = a[:, None]
        if r.ndim == 1:
            r = r[:, None]
        if t.ndim == 1:
            t = t[:, None]

        def pad_or_trim(x):
            T = x.shape[0]
            if T < self.context_len:
                pad = np.zeros((self.context_len - T, x.shape[1]), dtype=x.dtype)
                return np.concatenate([pad, x], axis=0)
            else:
                return x[-self.context_len:]

        o = pad_or_trim(o)
        a = pad_or_trim(a)
        r = pad_or_trim(r)
        t = pad_or_trim(t)

        return (
            torch.tensor(o, dtype=torch.float32),   # states
            torch.tensor(a, dtype=torch.float32),   # actions
            torch.tensor(r, dtype=torch.float32),   # returns
            torch.tensor(t, dtype=torch.long),      # timesteps
        )



# --- 学習ループ ---

#計画と行動のマルチタスクモデル
def _zscore_nonneg(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean()
    sd = x.std(unbiased=False) + eps
    z = (x - mu) / sd
    return torch.clamp(z, min=0.0)


#計画と行動のマルチタスクモデル
def train(pkl_path, context_len, embed_dim=128, n_layer=2, n_head=4, model_path=None):
#def train():

    dataset = SequenceDataset(pkl_path, context_len)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    for i, (states, actions, returns, timesteps) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  states.shape = {states.shape}")
        print(f"  actions.shape = {actions.shape}")
        print(f"  returns.shape = {returns.shape}")
        print(f"  timesteps.shape = {timesteps.shape}")
        break  # 一旦1バッチだけ確認

#計画と行動のマルチタスクモデル
    sample = dataset[0]
    states0, actions0, _, _, wp0, plan0 = sample
    obs_dim = states0.shape[-1]
    act_dim = actions0.shape[-1]
#   obs_dim = dataset[0][0].shape[-1]
#   act_dim = dataset[0][1].shape[-1]



## DTのMLP化検証
#    model = DecisionTransformer(obs_dim, act_dim).to(DEVICE)

## DTのMLP化検証 復元step1
#    model = DecisionTransformer(obs_dim, act_dim).to(DEVICE)

## DTのMLP化検証 復元step2
#    model = DecisionTransformer(obs_dim, act_dim).to(DEVICE)

## DTのMLP化検証 復元step3
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(DEVICE)

## DTのMLP化検証 復元step4
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(DEVICE)

## DTのMLP化検証 復元step5
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(DEVICE)

## DTのMLP化検証 復元step6
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len).to(DEVICE)

## DTのMLP化検証 復元step7
#    model = DecisionTransformer(obs_dim, act_dim,context_len=context_len, embed_dim=embed_dim).to(DEVICE)

#計画と行動のマルチタスクモデル
    model = DecisionTransformer(obs_dim, act_dim,
                                context_len=context_len,
                                embed_dim=embed_dim,
                                n_layer=n_layer,
                                n_head=n_head,
                                plan_M=PLAN_M,
                                use_focus=USE_FOCUS).to(DEVICE)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ モデルの重みを読み込みました: {model_path}")
    else:
       print(f"⚠️ モデルが存在しないため、新規で学習を開始します")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss()
#   model = DecisionTransformer(obs_dim, act_dim,context_len=context_len, embed_dim=embed_dim).to(DEVICE)
#
#   if os.path.exists(model_path):
#       model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#       print(f"✅ モデルの重みを読み込みました: {model_path}")
#   else:
#       print(f"⚠️ モデルが存在しないため、新規で学習を開始します")
#
#   optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#   loss_fn = nn.MSELoss()

    print("Start training...")

#計画と行動のマルチタスクモデル
    for epoch in range(EPOCHS):
        total_loss = 0
        for states, actions, returns, timesteps, wp, plan in dataloader:
            timesteps = timesteps.to(DEVICE).long()
            states    = states.to(DEVICE).float()
            actions   = actions.to(DEVICE).float()
            returns   = returns.to(DEVICE).float()
            wp        = wp.to(DEVICE).float()
            pred_actions, pred_plan, alpha = model(timesteps, states, actions, returns,
                                                    wp=wp, return_plan=True, return_focus=USE_FOCUS)
            # 行動損失（RTG重み付きBC）
            w = _zscore_nonneg(returns)  # (B,T,1)
            w = w.expand_as(actions)     # (B,T,2)
            L_act = (w * (pred_actions - actions) ** 2).mean()
            # 計画損失
            plan = plan.to(DEVICE).float()  # (B,2M)
            L_plan = mse(pred_plan, plan)
            # 滑らかさ損失（Δa）
            if pred_actions.shape[1] > 1:
                da = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
                L_smooth = (da ** 2).mean()
            else:
                L_smooth = torch.zeros((), device=DEVICE)
            loss = W_ACT * L_act + W_PLAN * L_plan + W_SMOOTH * L_smooth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            avg_loss = total_loss / max(1, len(dataloader))
            print(f"Epoch {epoch+1:03d} | L={avg_loss:.5f} | L_act={L_act.item():.4f} | L_plan={L_plan.item():.4f} | L_sm={L_smooth.item():.4f}")
#   for epoch in range(EPOCHS):
#       total_loss = 0
#       for states, actions, returns,timesteps in dataloader:
#           timesteps = timesteps.to(DEVICE)
#           states = states.to(DEVICE)
#           actions = actions.to(DEVICE)
#           returns = returns.to(DEVICE)
#
#           pred_actions = model(timesteps, states, actions, returns)
#           loss = loss_fn(pred_actions, actions)
#
#           optimizer.zero_grad()
#           loss.backward()
#           optimizer.step()
#
#           total_loss += loss.item()
#
#       avg_loss = total_loss / len(dataloader)
#       print(f"Epoch {epoch+1} - Loss: {avg_loss:.5f}")

    # 保存
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/decision_transformer.pt")
    print("Model saved to models/decision_transformer.pt")


if __name__ == "__main__":
    train(pkl_path, context_len,)
    
