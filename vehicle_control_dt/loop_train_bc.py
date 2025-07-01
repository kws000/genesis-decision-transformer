import subprocess
import time
import shutil

max_iterations = 20  # 無限でもいいがまずは安全に
for i in range(max_iterations):
    print(f"\n=== 🌀 Iteration {i+1} ===")

    # === 1. Expert data collection ===
    print("🚗 Running vehicle_control_drl.py to collect expert data...")
    subprocess.run(["python", "vehicle_control_drl.py"])

    # === 2. Check and back up the collected data (optional) ===
    expert_data_path = "expert_data/expert_data.csv"
    backup_path = f"expert_data_logs/expert_data_{i+1:03d}.csv"
    shutil.copy(expert_data_path, backup_path)

    # === 3. Train BC model ===
    print("🧠 Training bc_model from expert_data.csv...")
    subprocess.run(["python", "train_bc_model.py"])

    # === 4. Optionally save the model with a unique name ===
    shutil.copy("models/bc_model.pth", f"models/bc_model_{i+1:03d}.pth")

    # === 5. Optional sleep or cooldown ===
    time.sleep(5)

print("\n✅ All iterations completed.")
