from pathlib import Path
import numpy as np, torch

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

BASE = Path(__file__).resolve().parents[1]
DEF_MODEL_F = BASE/"models/vmax_factorized.pt"
DEF_SCALER_F = BASE/"models/scaler_factorized.npz"

def load_vmax_regressor_factorized(model_path=DEF_MODEL_F, scaler_path=DEF_SCALER_F):
    m = MLP(in_dim=3, hidden=128)
    m.load_state_dict(torch.load(str(model_path), map_location="cpu"), strict=True)
    m.eval()
    s = np.load(str(scaler_path), allow_pickle=True)
    expected = ["radius_m","mu","curvature_1pm"]
    x_cols = list(s["x_cols"])
    if x_cols != expected:
        raise ValueError(f"x_cols mismatch: {x_cols} vs {expected}")
    sc = {
        "x_mean": np.asarray(s["x_mean"], np.float32).reshape(1,-1),
        "x_std":  np.maximum(np.asarray(s["x_std"],  np.float32).reshape(1,-1), 1e-3),
        "y_mean": float(np.asarray(s["y_mean"])),
        "y_std":  float(np.asarray(s["y_std"])),
        "x_cols": np.array(x_cols),
    }
    return m, sc

def vmax_predict_factorized(m, sc, radius_m, mu, g=9.80665, safety=None):
    # safety 未指定なら学習時の平均値は不要（Aは外的因子として掛ける設計）
    r = max(float(radius_m), 1e-8)
    x = np.array([[r, float(mu), 1.0/r]], np.float32)
    xn = (x - sc["x_mean"]) / sc["x_std"]
    with torch.no_grad():
        y_base_n = m(torch.from_numpy(xn)).numpy()
    v_base = y_base_n * sc["y_std"] + sc["y_mean"]  # ≈ sqrt(mu*r)
    if safety is None:
        safety = 0.85  # 既定（必要なら外から渡す）
    v = float(v_base[0,0]) * (g**0.5) * float(safety)
    return v
