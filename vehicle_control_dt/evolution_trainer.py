import os
import subprocess
import time
import shutil
import re

#最新モデルでリプレイする　別手法
REPLAY_MODE = False#True#False
CHECKPOINTS_DIR = "checkpoints"

# ここで暫定モデル temp_model.pt がロードされる
TRY_CHECKPOINT_PATH = "checkpoints/temp_model.pt"
TRY_PKL_PATH = "data_dt/trajectories_dt.pkl"
TRY_NORM_PATH = "data_dt/mean_std.pkl"

# --- ハイパーパラメータステップ定義 ---
step_configs = [
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

        norm_path = f"checkpoints/step{step_id}_mean_std.pkl"
        pkl_path = f"checkpoints/step{step_id}_trajectories_dt.pkl"
        checkpoint_path = f"checkpoints/step{step_id}.pt"

        subprocess.run([
            "python", "evaluate_reward_once.py",
            "--context_len", str(config["context_len"]),
            "--n_layer", str(config["n_layer"]),
            "--n_head", str(config["n_head"]),
            #最新モデルでリプレイする　別手法
            "--norm_path",str(norm_path),
            "--pkl_path",str(pkl_path),
            "--checkpoint_path",str(checkpoint_path)
        ], encoding="utf-8")


def Evolution():

    # ----メイン----
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    stable_step = get_latest_stable_step()
    print(f"✅ 復元された安定ステップ: step{stable_step}")


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
        subprocess.run(["python", "expert_csv_to_pkl.py"])
        subprocess.run(["python", "convert_to_dt_format.py"])

        # --- 学習（失敗時中断） ---
        # ここで暫定モデル temp_model.pt が生成される
        result = subprocess.run([
            "python", "train_dt_external.py",
            "--context_len", str(config["context_len"]),
            "--n_layer", str(config["n_layer"]),
            "--n_head", str(config["n_head"]),
            #最新モデルでリプレイする　別手法
            "--norm_path",str(TRY_NORM_PATH),
            "--pkl_path",str(TRY_PKL_PATH),
            "--checkpoint_path",str(TRY_CHECKPOINT_PATH)
        ])
        if result.returncode != 0:
            print("❌ 学習エラーにより終了")
            break

        # --- 評価 ---
        print("=== 🧪 評価フェーズ ===")
        # ここで暫定モデルが評価される
        subprocess.run([
            "python", "evaluate_reward_once.py",
            "--context_len", str(config["context_len"]),
            "--n_layer", str(config["n_layer"]),
            "--n_head", str(config["n_head"]),
            #最新モデルでリプレイする　別手法
            "--norm_path",str(TRY_NORM_PATH),
            "--pkl_path",str(TRY_PKL_PATH),
            "--checkpoint_path",str(TRY_CHECKPOINT_PATH)
        ], encoding="utf-8")

        score = get_score()
        print(f"⭐ 評価スコア: {score:.2f}" if score > -float("inf") else "⚠️ 評価に失敗 or スコア不明")

        # リプレイ情報取得
        replay_start_waypoint_idx,replay_waypoint_direc = get_replay_info()

        # --- 判定と保存・復元 ---

    #前回スコアと比較    
        if score > prev_score:
    #    if score > -float("inf"):
            print("✅ 成長を確認。暫定モデルを確定して保存。")
            shutil.copy("checkpoints/temp_model.pt", f"{checkpoints_dir}/step{step_id}.pt")
    #    os.makedirs("models", exist_ok=True)
    #    torch.save(model.state_dict(), "models/decision_transformer.pt")

            # 正規化ファイルをstepX用として保存
            shutil.copy("data_dt/mean_std.pkl", f"{checkpoints_dir}/step{step_id}_mean_std.pkl")

            # 学習用データも同様に保存（必要であれば）
            shutil.copy("data_dt/trajectories_dt.pkl", f"{checkpoints_dir}/step{step_id}_trajectories_dt.pkl")


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

