# schema.py
OBS_COLS = [
    "target_wp_relative_x", "target_wp_relative_y",
    "pos_x", "pos_y",
    "yaw_sin", "yaw_cos",
    "velocity",
    "perp_error", "heading_error", "passed",
]
ACT_COLS = ["steer_angle", "throttle"]
PLAN_COLS = ["plan_x1","plan_y1","plan_x2","plan_y2","plan_x3","plan_y3"]
REWARD_COL = "reward"
