import os
import numpy as np
import pandas as pd
import pickle

# === 設定 ===
csv_path = "expert_data/expert_data.csv"
norm_path = "data_dt/mean_std.pkl"

# === CSVから観測ベクトルを生成 ===
df = pd.read_csv(csv_path)

# 変換後の観測ベクトル（9次元→10次元）
obs = np.stack([
    df["target_wp_x"],
    df["target_wp_y"],
    df["pos_x"],
    df["pos_y"],
    # 断続しないヨー角 １０次元に    
    df["yaw_sin"],
    df["yaw_cos"],
    df["velocity"],
    df["perp_error"],
    df["heading_error"],
    df["passed"],
], axis=1).astype(np.float32)

# === 統計量を計算 ===
new_mean = obs.mean(axis=0)
new_std = obs.std(axis=0) + 1e-6
new_count = obs.shape[0]

# === 前回統計量を読み込み（なければ初期化） ===
if os.path.exists(norm_path):
    with open(norm_path, "rb") as f:
        prev = pickle.load(f)
    prev_mean = prev["obs_mean"]
    prev_std = prev["obs_std"]
    prev_count = prev.get("count", 1)
else:
    prev_mean = np.zeros_like(new_mean)
    prev_std = np.ones_like(new_std)
    prev_count = 0

# === 平均：加重平均、標準偏差：最大値で統合 ===
total_count = prev_count + new_count
merged_mean = (prev_mean * prev_count + new_mean * new_count) / total_count
merged_std = np.maximum(prev_std, new_std)

# === 保存 ===
os.makedirs(os.path.dirname(norm_path), exist_ok=True)
with open(norm_path, "wb") as f:
    pickle.dump({
        "obs_mean": merged_mean,
        "obs_std": merged_std,
        "count": total_count,
    }, f)

print("✅ 統計量を更新・保存しました")
print("  mean:", merged_mean)
print("  std :", merged_std)
