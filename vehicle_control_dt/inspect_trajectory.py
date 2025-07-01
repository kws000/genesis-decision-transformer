# vehicle_control_dt/inspect_trajectory.py

import pickle

with open("trajectories/trajectory_data.pkl", "rb") as f:
    data = pickle.load(f)

print("データ数:", len(data))
print("最初のデータのキー:", data[0].keys())
print("最初のデータの内容例:")
for k, v in data[0].items():
    print(f"{k}: shape={getattr(v, 'shape', type(v))}")
