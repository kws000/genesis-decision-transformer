import os
import pickle
import numpy as np
import torch

from world_training.laws.world_mlp import WorldMLP


OBS_V2_KEYS = [
    "target_x", "target_y",
    "pos_x", "pos_y",
    "yaw_sin", "yaw_cos",
    "velocity",
    "perp_error",
    "heading_error",
    "passed",
    "kappa_local",
    "mu_local",
    "vmax_local",
    "v_ratio",
    "headroom",
    "vmax_min_hH",
    "vmax_mean_hH",
    "vmax_slope_hH",
    "limit_v_target",
]


IDX = {k: i for i, k in enumerate(OBS_V2_KEYS)}


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_mean_std(mean_std_path):
    obj = _load_pickle(mean_std_path)

    if isinstance(obj, dict):
        if "obs_mean" in obj and "obs_std" in obj:
            obs_mean = obj["obs_mean"]
            obs_std = obj["obs_std"]
        elif "mean" in obj and "std" in obj:
            obs_mean = obj["mean"]
            obs_std = obj["std"]
        elif "state_mean" in obj and "state_std" in obj:
            obs_mean = obj["state_mean"]
            obs_std = obj["state_std"]
        else:
            raise RuntimeError(f"Unknown mean/std keys: {list(obj.keys())}")
    else:
        raise RuntimeError(f"Unsupported mean/std format: {type(obj)}")

    obs_mean = np.asarray(obs_mean, dtype=np.float32).reshape(-1)
    obs_std = np.asarray(obs_std, dtype=np.float32).reshape(-1)
    obs_std = np.where(np.abs(obs_std) < 1e-8, 1.0, obs_std)

    return obs_mean, obs_std

#損失に世界モデルを使う　可視化　#テンソル構造が逆？
def denorm_obs(obs_norm, obs_mean, obs_std):
    obs_norm = np.asarray(obs_norm, dtype=np.float32)

    # (1, 19) や (1, 1, 19) で来ても (19,) に潰す
    obs_norm = obs_norm.reshape(-1)

    if obs_norm.shape[0] != 19:
        raise RuntimeError(f"obs_norm must flatten to 19 dims, got {obs_norm.shape}")

    return obs_norm * obs_std + obs_mean
#def denorm_obs(obs_norm, obs_mean, obs_std):
#    obs_norm = np.asarray(obs_norm, dtype=np.float32)
#    return obs_norm * obs_std + obs_mean

#損失に世界モデルを使う　可視化　#テンソル構造が逆？
def norm_obs(obs_phys, obs_mean, obs_std):
    obs_phys = np.asarray(obs_phys, dtype=np.float32)
    obs_phys = obs_phys.reshape(-1)

    if obs_phys.shape[0] != 19:
        raise RuntimeError(f"obs_phys must flatten to 19 dims, got {obs_phys.shape}")

    return (obs_phys - obs_mean) / obs_std
#def norm_obs(obs_phys, obs_mean, obs_std):
#    obs_phys = np.asarray(obs_phys, dtype=np.float32)
#    return (obs_phys - obs_mean) / obs_std



class WorldModelRuntime:
    """
    Runtime wrapper for World MLP.

    Input:
        obs_norm    : (19,)
        prev_action : (2,)
        action      : (2,)

    Output:
        next_obs_norm : (19,)
        delta_norm    : (19,)

    Note:
        velocity / v_ratio are currently reference only.
        Main trusted debug targets are:
            perp_error
            heading_error
    """

    def __init__(
        self,
        model_path,
        mean_std_path,
        device=None,
    ):
        self.model_path = model_path
        self.mean_std_path = mean_std_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.obs_mean, self.obs_std = load_mean_std(mean_std_path)

        ckpt = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )

        input_dim = ckpt.get("input_dim", 23)
        output_dim = ckpt.get("output_dim", 19)
        hidden_dim = ckpt.get("hidden_dim", 128)

        self.model = WorldMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if self.input_dim != 23:
            raise RuntimeError(f"Unexpected input_dim: {self.input_dim}")

        if self.output_dim != 19:
            raise RuntimeError(f"Unexpected output_dim: {self.output_dim}")

    @torch.no_grad()
    def predict_next_norm(self, obs_norm, prev_action, action):
        obs_norm = np.asarray(obs_norm, dtype=np.float32).reshape(-1)
        prev_action = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if obs_norm.shape[0] != 19:
            raise RuntimeError(f"obs_norm must be (19,), got {obs_norm.shape}")

        if prev_action.shape[0] != 2:
            raise RuntimeError(f"prev_action must be (2,), got {prev_action.shape}")

        if action.shape[0] != 2:
            raise RuntimeError(f"action must be (2,), got {action.shape}")

        x = np.concatenate(
            [
                obs_norm,
                prev_action,
                action,
            ],
            axis=0,
        ).astype(np.float32)

        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)

        delta_norm = self.model(x_t).squeeze(0).cpu().numpy()
        next_obs_norm = obs_norm + delta_norm

        return next_obs_norm, delta_norm

    @torch.no_grad()
    def predict_next_phys(self, obs_norm, prev_action, action):
        next_obs_norm, delta_norm = self.predict_next_norm(
            obs_norm=obs_norm,
            prev_action=prev_action,
            action=action,
        )

        obs_phys = denorm_obs(obs_norm, self.obs_mean, self.obs_std)
        next_obs_phys = denorm_obs(next_obs_norm, self.obs_mean, self.obs_std)

        return next_obs_phys, next_obs_norm, delta_norm, obs_phys

    def debug_dict(self, obs_norm, prev_action, action):

        #contextlen=1以上だと落ちる
        if obs_norm.ndim >= 2:
            obs_norm_single = obs_norm.reshape(-1, 19)[-1]
        else:
            obs_norm_single = obs_norm.reshape(-1)
            if obs_norm_single.shape[0] > 19 and obs_norm_single.shape[0] % 19 == 0:
                obs_norm_single = obs_norm_single.reshape(-1, 19)[-1]

        if prev_action.ndim >= 2:
            prev_action_single = prev_action.reshape(-1, 2)[-1]
        else:
            prev_action_single = prev_action.reshape(-1)
            if prev_action_single.shape[0] > 2 and prev_action_single.shape[0] % 2 == 0:
                prev_action_single = prev_action_single.reshape(-1, 2)[-1]

        if action.ndim >= 2:
            action_single = action.reshape(-1, 2)[-1]
        else:
            action_single = action.reshape(-1)
            if action_single.shape[0] > 2 and action_single.shape[0] % 2 == 0:
                action_single = action_single.reshape(-1, 2)[-1]

        next_phys, next_norm, delta_norm, obs_phys = self.predict_next_phys(
            obs_norm=obs_norm_single,
            prev_action=prev_action_single,
            action=action_single,
        )

        d = {
            "perp_now": float(obs_phys[IDX["perp_error"]]),
            "heading_now": float(obs_phys[IDX["heading_error"]]),
            "velocity_now": float(obs_phys[IDX["velocity"]]),
            "v_ratio_now": float(obs_phys[IDX["v_ratio"]]),
            "limit_v_target_now": float(obs_phys[IDX["limit_v_target"]]),

            "wm_perp_next": float(next_phys[IDX["perp_error"]]),
            "wm_heading_next": float(next_phys[IDX["heading_error"]]),
            "wm_velocity_next_ref": float(next_phys[IDX["velocity"]]),
            "wm_v_ratio_next_ref": float(next_phys[IDX["v_ratio"]]),
            "wm_limit_v_target_next": float(next_phys[IDX["limit_v_target"]]),

            "wm_delta_perp_norm": float(delta_norm[IDX["perp_error"]]),
            "wm_delta_heading_norm": float(delta_norm[IDX["heading_error"]]),
            "wm_delta_velocity_norm_ref": float(delta_norm[IDX["velocity"]]),
            "wm_delta_v_ratio_norm_ref": float(delta_norm[IDX["v_ratio"]]),
        }

        return d

    def debug_line(self, obs_norm, prev_action, action, prefix="[WM]"):

        #contextlen=1以上だと落ちる
        if obs_norm.ndim >= 2:
            obs_norm_single = obs_norm.reshape(-1, 19)[-1]
        else:
            obs_norm_single = obs_norm.reshape(-1)
            if obs_norm_single.shape[0] > 19 and obs_norm_single.shape[0] % 19 == 0:
                obs_norm_single = obs_norm_single.reshape(-1, 19)[-1]

        if prev_action.ndim >= 2:
            prev_action_single = prev_action.reshape(-1, 2)[-1]
        else:
            prev_action_single = prev_action.reshape(-1)
            if prev_action_single.shape[0] > 2 and prev_action_single.shape[0] % 2 == 0:
                prev_action_single = prev_action_single.reshape(-1, 2)[-1]

        if action.ndim >= 2:
            action_single = action.reshape(-1, 2)[-1]
        else:
            action_single = action.reshape(-1)
            if action_single.shape[0] > 2 and action_single.shape[0] % 2 == 0:
                action_single = action_single.reshape(-1, 2)[-1]

        d = self.debug_dict(
            obs_norm=obs_norm_single,
            prev_action=prev_action_single,
            action=action_single,
        )

        return (
            f"{prefix} "
            f"perp {d['perp_now']:+.4f}->{d['wm_perp_next']:+.4f}  "
            f"head {d['heading_now']:+.4f}->{d['wm_heading_next']:+.4f}  "
            f"vel(ref) {d['velocity_now']:+.3f}->{d['wm_velocity_next_ref']:+.3f}  "
            f"vr(ref) {d['v_ratio_now']:+.3f}->{d['wm_v_ratio_next_ref']:+.3f}"
        )


def default_paths_from_project(project_root):
    """
    project_root:
        .../vehicle_control_dt
    """
    model_path = os.path.join(
        project_root,
        "world_training",
        "models",
        "world_mlp.pt",
    )

    mean_std_path = os.path.join(
        project_root,
        "data_dt",
        "base_mean_std.pkl",
    )

    return model_path, mean_std_path


if __name__ == "__main__":
    # simple smoke test
    this_file = os.path.abspath(__file__)
    laws_dir = os.path.dirname(this_file)
    world_training_dir = os.path.dirname(laws_dir)
    project_root = os.path.dirname(world_training_dir)

    model_path, mean_std_path = default_paths_from_project(project_root)

    print("[SMOKE TEST]")
    print(f"project_root : {project_root}")
    print(f"model_path   : {model_path}")
    print(f"mean_std_path: {mean_std_path}")

    wm = WorldModelRuntime(
        model_path=model_path,
        mean_std_path=mean_std_path,
    )

    obs_norm = np.zeros(19, dtype=np.float32)
    prev_action = np.zeros(2, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)

    print(wm.debug_line(obs_norm, prev_action, action))