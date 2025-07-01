# loop_train_dt.py

import subprocess
import time

while True:
    print("\n=== Step 1: Run vehicle_control_drl ===")
    subprocess.run(["python", "vehicle_control_drl.py"])

    print("\n=== Step 2: Convert expert_data.csv to trajectory_data.pkl ===")
    subprocess.run(["python", "expert_csv_to_pkl.py"])

    print("\n=== Step 3: Convert trajectory_data.pkl to DT format ===")
    subprocess.run(["python", "convert_to_dt_format.py"])

    print("\n=== Step 4: Train Decision Transformer ===")
    subprocess.run(["python", "train_dt.py"])

    print("\n✅ One training cycle completed. Sleeping for 5 seconds...\n")
    time.sleep(5)
