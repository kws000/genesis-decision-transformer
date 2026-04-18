import os
import subprocess
import time
import shutil
import re

#下記に保存されているステップを指定
#C:\Users\kws00\Genesis4D\examples\tutorials\vehicle_control_dt\checkpoints

PREVIEW_STEP_ID = 6

#進化ループの大改修	正規化の固定統計
BASE_NORM_PKL = "data_dt/base_mean_std.pkl"   # ★固定統計

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

def Preview():
        
    step_id = PREVIEW_STEP_ID

    viz_out = f"viz/step{step_id}"

    step_config = step_configs[step_id]

    subprocess.run([
        "python", "visualize_dt_attention.py",
        "--checkpoint",    f"checkpoints/step{step_id}.pt",
        "--pkl",           f"checkpoints/step{step_id}_trajectories_dt.pkl",
        #進化ループの大改修	正規化の固定統計
        "--norm_path",     BASE_NORM_PKL,
        "--context_len",   str(step_config["context_len"]),
        "--n_layer",       str(step_config["n_layer"]),
        "--n_head",        str(step_config["n_head"]),
        "--embed_dim",     "128",
        "--outdir",        viz_out,
        "--sample_index",    "301",
        # "--obs_names",   "x,y,..."  # 任意
    ], check=False)



def main():

    Preview()

if __name__ == "__main__":
    main()

