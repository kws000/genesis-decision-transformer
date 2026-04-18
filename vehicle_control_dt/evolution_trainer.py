import os
import subprocess
import time
import shutil
import re

import os, glob, shutil, time, json
from pathlib import Path

import json
import os

import random
from typing import Optional, Iterable


#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 経過確認ログ
from debug_probes import probe_csv,probe_raw_pkl,probe_dt


#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 リセット機構
ROOT = Path(__file__).resolve().parent
DATA_DIRS = [
    ROOT/"expert_data",         # CSV
    ROOT/"trajectories",        # raw PKL
#進化ループの大改修 cleanはもう封印だと思う
#    ROOT/"data_dt",             # DT-PKL / mean_std
]
CKPT_DIR  = ROOT/"checkpoints"  # モデル
TMP_FILES = [
    ROOT/"checkpoints/temp_model.pt",
    ROOT/"eval_score.txt",
    ROOT/"replay_info.txt",
]


#最新モデルでリプレイする　別手法
REPLAY_MODE = False#True#False
CHECKPOINTS_DIR = "checkpoints"

# ここで暫定モデル temp_model.pt がロードされる
TRY_CHECKPOINT_PATH = "checkpoints/temp_model.pt"

#進化ループの大改修	正規化の固定統計
BASE_NORM_PKL = "data_dt/base_mean_std.pkl"   # ★固定統計

# --- ハイパーパラメータステップ定義 ---

#進化ループの大改修 低学年では全員合格
BABY_STEP = 3

step_configs = [
    #進化ループの大改修 低学年では全員合格
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},

    # ここからが本番で試験あり
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},
    {"context_len": 1, "n_layer": 3, "n_head": 4},

    {"context_len": 2, "n_layer": 3, "n_head": 4},
    {"context_len": 2, "n_layer": 3, "n_head": 4},
    {"context_len": 2, "n_layer": 3, "n_head": 4},
    {"context_len": 2, "n_layer": 3, "n_head": 4},

    {"context_len": 3, "n_layer": 3, "n_head": 4},
    {"context_len": 3, "n_layer": 3, "n_head": 4},
    {"context_len": 3, "n_layer": 3, "n_head": 4},
    {"context_len": 3, "n_layer": 3, "n_head": 4},

    {"context_len": 4, "n_layer": 3, "n_head": 4},
    {"context_len": 4, "n_layer": 3, "n_head": 4},
    {"context_len": 4, "n_layer": 3, "n_head": 4},

    {"context_len": 5, "n_layer": 3, "n_head": 4},
    {"context_len": 5, "n_layer": 3, "n_head": 4},
    {"context_len": 5, "n_layer": 3, "n_head": 4},

    {"context_len": 6, "n_layer": 3, "n_head": 4},
    {"context_len": 6, "n_layer": 3, "n_head": 4},

    {"context_len": 7, "n_layer": 3, "n_head": 4},
    {"context_len": 7, "n_layer": 3, "n_head": 4},
    
    {"context_len": 8, "n_layer": 3, "n_head": 4},
    {"context_len": 9, "n_layer": 3, "n_head": 4},
    {"context_len": 10, "n_layer": 3, "n_head": 4},
    {"context_len": 11, "n_layer": 3, "n_head": 4},
    {"context_len": 12, "n_layer": 3, "n_head": 4},
    {"context_len": 13, "n_layer": 3, "n_head": 4},
    {"context_len": 14, "n_layer": 3, "n_head": 4},
]

# ----関数----


#進化ループの大改修 ds_blender抽選の仕組み
def _ds_id_num(ds_id: str) -> int:
    # ds_000123 -> 123
    return int(ds_id.split("_")[1])

#進化ループの大改修 ds_blender抽選の仕組み
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#進化ループの大改修 ds_blender抽選の仕組み
def _save_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

#進化ループの大改修 ds_blender抽選の仕組み
#def resolve_best_ds(checkpoints_dir: str, default_ds: str) -> str:
def resolve_anchor_latest(root_dir="data_dt"):
    col = _load_json(os.path.join(root_dir, "ds_collection.json"))
    ds_ids = [s["ds_id"] for s in col.get("snapshots", [])]
    if not ds_ids:
        raise RuntimeError("ds_collection.json has no snapshots")
    ds_ids_sorted = sorted(ds_ids, key=_ds_id_num)
    anchor = ds_ids_sorted[0]
    latest = ds_ids_sorted[-1]
    return anchor, latest


#進化ループの大改修 ds_blender抽選を正しく行うように
def resolve_shuffle_ds(
    root_dir: str = "data_dt",
    default_ds: Optional[str] = None,
    exclude: Optional[Iterable[str]] = None,
    progress_vs_clean_prob: float = 0.5,
) -> str:
    """
    ds_collection.json を元に、progress系 or clean系 のどちらかの上位ソート列から
    weighted lottery で1つ ds_id を選ぶ。

    仕様:
      - progress 上位リスト作成
      - clean 上位リスト作成
      - 最初に progress / clean のどちらの軸で選ぶか抽選
      - 選ばれたリスト内で、上位ほど高確率
      - 先頭重み = 25%, 末尾重み = 2.5%（相対比 10:1）
      - exclude に入っている ds_id は候補から除外
      - 候補が無ければ default_ds を返す

    Args:
        root_dir: data_dt ルート
        default_ds: 候補が空のとき返す fallback
        exclude: 除外したい ds_id 群
        progress_vs_clean_prob:
            progress リストを選ぶ確率（残りは clean）
            0.5 なら 50:50

    Returns:
        選ばれた ds_id
    """
    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)

    coll_path = os.path.join(root_dir, "ds_collection.json")
    if not os.path.isfile(coll_path):
        if default_ds is None:
            raise FileNotFoundError(f"ds_collection.json not found: {coll_path}")
        return default_ds

    with open(coll_path, "r", encoding="utf-8") as f:
        coll = json.load(f)

    snapshots = coll.get("snapshots", [])
    if not snapshots:
        if default_ds is None:
            raise RuntimeError("No snapshots found in ds_collection.json")
        return default_ds

    # 候補抽出
    candidates = []
    for snap in snapshots:
        ds_id = snap.get("ds_id")
        if not ds_id or ds_id in exclude:
            continue

        summary = snap.get("summary", {}) or {}
        rtg_prog = float(summary.get("rtg_prog", 0.0) or 0.0)
        rtg_clean = float(summary.get("rtg_clean", 0.0) or 0.0)

        candidates.append({
            "ds_id": ds_id,
            "rtg_prog": rtg_prog,
            "rtg_clean": rtg_clean,
        })

    if not candidates:
        return default_ds if default_ds is not None else ""

    # progress系 / clean系 を最初に抽選
    use_progress = (random.random() < progress_vs_clean_prob)

    if use_progress:
        ranked = sorted(
            candidates,
            key=lambda x: (x["rtg_prog"], x["rtg_clean"], x["ds_id"]),
            reverse=True
        )
    else:
        ranked = sorted(
            candidates,
            key=lambda x: (x["rtg_clean"], x["rtg_prog"], x["ds_id"]),
            reverse=True
        )

    n = len(ranked)

    # ランク重み:
    # 先頭 0.25, 末尾 0.025 になるように線形補間
    # n=1 のときは先頭重みのみ
    top_w = 0.25
    bottom_w = 0.025

    if n == 1:
        weights = [top_w]
    else:
        weights = []
        for i in range(n):
            t = i / (n - 1)   # 0.0 (先頭) -> 1.0 (末尾)
            w = top_w + (bottom_w - top_w) * t
            weights.append(w)

    chosen = random.choices(ranked, weights=weights, k=1)[0]
    return chosen["ds_id"]

#進化ループの大改修 ds_blender抽選の仕組み
def resolve_best_ds(checkpoints_dir: str, default_ds: str) -> str:
    """
    最短：最新の step*_ds.txt を best とみなす。
    （本当は「ベストモデルのstep」を使うが、まず動くv1として）
    """
    best = None
    if os.path.isdir(checkpoints_dir):
        cand = []
        for fn in os.listdir(checkpoints_dir):
            if fn.startswith("step") and fn.endswith("_ds.txt"):
                # step12_ds.txt -> 12
                try:
                    step = int(fn[len("step"):].split("_")[0])
                    cand.append((step, fn))
                except:
                    pass
        if cand:
            cand.sort()
            best_fn = cand[-1][1]
            with open(os.path.join(checkpoints_dir, best_fn), "r", encoding="utf-8") as f:
                best = f.read().strip()
    return best or default_ds

#進化ループの大改修 ds_blender抽選の仕組み
def write_ds_blender_v1(root_dir="data_dt", checkpoints_dir="checkpoints"):
    anchor, latest = resolve_anchor_latest(root_dir)


#進化ループの大改修 ds_blender抽選を正しく行うように
    best = resolve_shuffle_ds(root_dir="data_dt",default_ds=latest,exclude={anchor, latest})
#    best = resolve_best_ds(checkpoints_dir, default_ds=latest)

    # ルールベースv1
    w = {
        best:   0.60,
        latest: 0.25,
#        anchor: 0.15,
    }
    # 同一dsは合算されるのでOK
    items = [{"ds_id": k, "weight": float(v)} for k, v in w.items()]
    # 正規化
    s = sum(x["weight"] for x in items)
    for x in items:
        x["weight"] /= max(1e-9, s)

    blender = {
        "version": 2,
        "snapshot_mix": items,
        "binning": {
            "prog_fast_thr": 0.75,
            "clean_safe_thr": 0.75,
            "bins": ["safe", "fast", "boundary", "both"]
        },
        "bin_mix": {
            "safe": 0.20,
            "fast": 0.20,
            "boundary": 0.50,
            "both": 0.10
        },
        "sampling": {
            "snapshot_pick": "categorical",
            "episode_pick": "uniform",
            "window_pick": "uniform"
        }
    }

    out = os.path.join(root_dir, "ds_blender.json")
    _save_json(out, blender)
    print(f"✅ ds_blender.json updated (v1): best={best}, latest={latest}, anchor={anchor}")


#進化ループの大改修
def get_active_snapshot_paths(root_dir="data_dt"):
    import os, json

    blender_path = os.path.join(root_dir, "ds_blender.json")
    collection_path = os.path.join(root_dir, "ds_collection.json")

    with open(blender_path, "r", encoding="utf-8") as f:
        bl = json.load(f)

    ds_id = bl["snapshot_mix"][0]["ds_id"]

    with open(collection_path, "r", encoding="utf-8") as f:
        col = json.load(f)

    path_map = {s["ds_id"]: s.get("path", f"snapshots/{s['ds_id']}") 
                for s in col["snapshots"]}

    snap_dir = os.path.join(root_dir, path_map[ds_id])

    return {
        "ds_id": ds_id,
        "traj": os.path.join(snap_dir, "trajectories_dt.pkl")
    }


#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 リセット機構
def _rm(path: Path):
    try:
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        print(f"[CLEAN] skip {path}: {e}")

#ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 リセット機構
def clean_all_intermediates():
    print("[CLEAN] removing old intermediates …")
    for d in DATA_DIRS:
        _rm(d)
    for f in TMP_FILES:
        _rm(f)
    # checkpoints/ は消しすぎ注意：ステップ0の時だけ全部消す
    if CKPT_DIR.exists():
        for f in CKPT_DIR.glob("step*.pt"):
            f.unlink(missing_ok=True)
    print("[CLEAN] done.")


# === 安定ステップを自動判定 ===
def get_latest_stable_step():
    step_files = [f for f in os.listdir(CHECKPOINTS_DIR) if re.match(r"step(\d+)\.pt", f)]

    step_ids = []
    for f in step_files:
        match = re.match(r"step(\d+)\.pt", f)
        if match:
            step_ids.append(int(match.group(1)))

    return max(step_ids) if step_ids else -1

# === 評価スコアの取得ヘルパー ===
def get_score():
    try:
        with open("eval_score.txt", "r") as f:
            return float(f.read().strip())
    except Exception as e:
        print(f"⚠️ 評価スコア読み込み失敗: {e}")
        return -float("inf")

# === リプレイ情報取得ヘルパー ===
def get_replay_info():
    try:
        with open("replay_info.txt", "r") as f:
            return int(f.readline().strip()),int(f.readline().strip())
    except Exception as e:
        print(f"⚠️ リプレイ情報読み込み失敗: {e}")
        return -int("0"),-int("0")

def Replay():

    # ----メイン----
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    stable_step = get_latest_stable_step()
    print(f"✅ 最終ステップのリプレイ: step{stable_step}")
    
    if stable_step >= 0 and stable_step < len(step_configs):

        step_id = stable_step

        config = step_configs[step_id]

        print(f"\n=== 🚀 Step {step_id}: config={config} ===")

#進化ループの大改修	推論側        
        subprocess.run([
            "python", "evaluate_reward_once.py",
            "--context_len", str(config["context_len"]),
            "--n_layer", str(config["n_layer"]),
            "--n_head", str(config["n_head"]),
            "--checkpoint_path", str(TRY_CHECKPOINT_PATH)
#外だしする必要はない筈
#            "--ds_id", "auto"   # 最短：ds_blender の先頭dsを使う
        ], encoding="utf-8")
#        subprocess.run([
#            "python", "evaluate_reward_once.py",
#            "--context_len", str(config["context_len"]),
#            "--n_layer", str(config["n_layer"]),
#            "--n_head", str(config["n_head"]),
#            #最新モデルでリプレイする　別手法
#            "--norm_path",str(norm_path),
#            "--pkl_path",str(pkl_path),
#            "--checkpoint_path",str(checkpoint_path)
#        ], encoding="utf-8")


def Evolution():

    # ----メイン----
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    stable_step = get_latest_stable_step()
    print(f"✅ 復元された安定ステップ: step{stable_step}")

    #ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 リセット機構
    if stable_step < 0:
        print(f"✅ 初回なので、全ての中間ファイルを削除")
        clean_all_intermediates()
        #フォルダも消してる
        os.makedirs("expert_data", exist_ok=True)


    # 前回までのスコアファイルと最終スコア値
    prev_score = 0
    if stable_step >= 0:
        pre_score_path = f"checkpoints/step{stable_step}_score.txt"
        try:
            with open(pre_score_path, "r") as f:
                prev_score = float(f.read().strip())
        except Exception as e:
            prev_score = 0

    # === 進化ループ ===
    step_id = stable_step + 1
    while step_id < len(step_configs):
    #for step_id in range(stable_step + 1, len(step_configs)):

        config = step_configs[step_id]
        print(f"\n=== 🚀 Step {step_id}: config={config} ===")

        # --- データ生成と変換 ---
        subprocess.run(["python", "vehicle_control_drl.py"])

        #ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 経過確認ログ run_control_loop(...) 等が終わった直後
        probe_csv("expert_data/expert_data.csv")

        subprocess.run(["python", "expert_csv_to_pkl.py"])

        #ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 経過確認ログ export_csv_to_pkl.py を呼んだ直後
        probe_raw_pkl("trajectories/trajectory_data.pkl")
			
        subprocess.run(["python", "convert_to_dt_format.py"])

		#進化ループの大改修	ds_blender抽選の仕組み
        write_ds_blender_v1(root_dir="data_dt", checkpoints_dir=checkpoints_dir)

        #ボトルネック認識とVmax魂の注入 アクセルが小さすぎる問題 経過確認ログ convert_to_dt_format.py を呼んだ直後
#進化ループの大改修 今の仕組みと合わないので削除        
#        probe_dt("data_dt/trajectories_dt.pkl", "data_dt/mean_std.pkl")			

        # --- 学習（失敗時中断） ---
        # ここで暫定モデル temp_model.pt が生成される

#進化ループの大改修 新しいフォルダ構成
        result = subprocess.run([
            "python", "train_dt_external.py",
            "--context_len", str(config["context_len"]),
            "--n_layer", str(config["n_layer"]),
            "--n_head", str(config["n_head"]),
            "--checkpoint_path", str(TRY_CHECKPOINT_PATH)
        ])        
#        result = subprocess.run([
#            "python", "train_dt_external.py",
#            "--context_len", str(config["context_len"]),
#            "--n_layer", str(config["n_layer"]),
#            "--n_head", str(config["n_head"]),
#            #最新モデルでリプレイする　別手法
#            "--norm_path",str(TRY_NORM_PATH),
#            "--pkl_path",str(TRY_PKL_PATH),
#            "--checkpoint_path",str(TRY_CHECKPOINT_PATH)
#        ])

        if result.returncode != 0:
            print("❌ 学習エラーにより終了")
            break

        # --- 評価 ---
        print("=== 🧪 評価フェーズ ===")
        # ここで暫定モデルが評価される
#進化ループの大改修	推論側        
        subprocess.run([
            "python", "evaluate_reward_once.py",
            "--context_len", str(config["context_len"]),
            "--n_layer", str(config["n_layer"]),
            "--n_head", str(config["n_head"]),
            "--checkpoint_path", str(TRY_CHECKPOINT_PATH)
#外だしする必要はない筈
#            "--ds_id", "auto"   # 最短：ds_blender の先頭dsを使う
        ], encoding="utf-8")
#        subprocess.run([
#            "python", "evaluate_reward_once.py",
#            "--context_len", str(config["context_len"]),
#            "--n_layer", str(config["n_layer"]),
#            "--n_head", str(config["n_head"]),
#            #最新モデルでリプレイする　別手法
#            "--norm_path",str(TRY_NORM_PATH),
#            "--pkl_path",str(TRY_PKL_PATH),
#            "--checkpoint_path",str(TRY_CHECKPOINT_PATH)
#        ], encoding="utf-8")

        score = get_score()
        print(f"⭐ 評価スコア: {score:.2f}" if score > -float("inf") else "⚠️ 評価に失敗 or スコア不明")

        # リプレイ情報取得
        replay_start_waypoint_idx,replay_waypoint_direc = get_replay_info()

        # --- 判定と保存・復元 ---

    #前回スコアと比較    
#進化ループの大改修 低学年では全員合格
# 近いスコアを捨てるのは勿体ない、、少しマージンを与えよう
        if (score > prev_score * 0.9) or (step_id < BABY_STEP):
            score = 0 if score < 0 else score
            # マージンを反映し続けると、どんどんスコアが下がっていくだけなので、前回スコアにしておく
            score = score if score > prev_score else prev_score
#        if score > prev_score:
            print("✅ 成長を確認。暫定モデルを確定して保存。")
            shutil.copy("checkpoints/temp_model.pt", f"{checkpoints_dir}/step{step_id}.pt")

            # 正規化ファイルをstepX用として保存

#進化ループの大改修
            paths = get_active_snapshot_paths()
            shutil.copy(paths["traj"],     f"{checkpoints_dir}/step{step_id}_trajectories_dt.pkl")
#            shutil.copy("data_dt/mean_std.pkl", f"{checkpoints_dir}/step{step_id}_mean_std.pkl")
#            shutil.copy("data_dt/trajectories_dt.pkl", f"{checkpoints_dir}/step{step_id}_trajectories_dt.pkl")


            stable_step = step_id

            #確定モデルのスコアファイル出力
            score_path = f"checkpoints/step{stable_step}_score.txt"
            with open(score_path, "w") as f:
                f.write(f"{score:.2f}")

            replay_path = f"checkpoints/step{stable_step}_replay.txt"
            with open(replay_path, "w") as f:
                f.write(f"{replay_start_waypoint_idx}"+'\n')
                f.write(f"{replay_waypoint_direc}"+'\n')

            #最大スコア更新
            prev_score = score
            #次のステップへ
            step_id += 1
        else:
    # 暫定モデルを破棄するだけ        
            print("❌ スコア悪化 or 評価失敗。暫定モデルは破棄します。")
    #        print("❌ スコア悪化 or 評価失敗。ロールバックします。")

    # 何もしなくていい
    #        if stable_step >= 0:
    #            print(f"↩️ 復元: step{stable_step}.pt")
    #            shutil.copy(f"{checkpoints_dir}/step{stable_step}.pt", "checkpoints/temp_model.pt")
    #        else:
    #            print("⚠️ 初期ステップのため復元不能。")

        print("✅ One evolution step completed. Sleeping...\n")
        time.sleep(5)

def main():

    #最新モデルでリプレイする　別手法
    if REPLAY_MODE:
        Replay()
    else:
        Evolution()

if __name__ == "__main__":
    main()

