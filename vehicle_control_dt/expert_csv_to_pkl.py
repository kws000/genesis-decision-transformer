# csv_to_pkl.py（collect_trajectory 形式に修正）

import pandas as pd
import pickle
import numpy as np
import os

csv_path = "expert_data/expert_data.csv"
output_path = "trajectories/trajectory_data.pkl"

df = pd.read_csv(csv_path)

# obs = 状況ベクトル

# 断続しないヨー角 １０次元に
obs = df[["target_wp_x", "target_wp_y", "pos_x", "pos_y", "yaw_sin", "yaw_cos", "velocity",
          "perp_error", "heading_error", "passed"]].values.astype(np.float32)
#obs = df[["target_wp_x", "target_wp_y", "pos_x", "pos_y", "yaw", "velocity",
#          "perp_error", "heading_error", "passed"]].values.astype(np.float32)

# act = 出力ベクトル
act = df[["steer_angle", "throttle"]].values.astype(np.float32)

# reward = heading_error の符号付き距離（任意の定義に応じて調整）
if "reward" in df.columns:
    rew = df["reward"].values.astype(np.float32)
else:
    rew = -np.abs(df["heading_error"].values).astype(np.float32)

# next_obs
next_obs = np.roll(obs, -1, axis=0)
next_obs[-1] = obs[-1]

# done
done = np.zeros(len(obs), dtype=bool)
done[-1] = True

# pack as one trajectory dictionary
trajectory = {
    "obs": obs,
    "action": act,
    "reward": rew,
    "done": done,
    "next_obs": next_obs,
}

# save as list of one trajectory
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump([trajectory], f)

print(f"✅ Saved in collect_trajectory format: {output_path}")
