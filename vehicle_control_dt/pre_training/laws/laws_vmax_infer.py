from pathlib import Path
import numpy as np, torch
import math

#下記は座学用で、以前やったControlMLP(ステアとアクセルの模倣学習)とは別物なので注意
class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

#ボトルネック認識とVmax魂の注入
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

    #vmaxモデルの符号バグ ★追加: x_min/x_max（古いscalerは無いので安全に）
    if "x_min" in s.files and "x_max" in s.files:
        sc["x_min"] = np.asarray(s["x_min"], np.float32).reshape(1,-1)
        sc["x_max"] = np.asarray(s["x_max"], np.float32).reshape(1,-1)
    else:
        sc["x_min"] = None
        sc["x_max"] = None
     
    return m, sc

#vmaxモデルの符号バグ
def _v_phys(radius_m: float, mu: float, g: float, safety: float) -> float:
    r = max(float(radius_m), 1e-8)
    m = max(float(mu), 0.0)
    return float(safety) * math.sqrt(m * float(g) * r)

#vmaxモデルの符号バグ
def vmax_predict_factorized(m, sc, radius_m, mu, g=9.80665, safety=None,
                            xn_abs_max: float = 10.0) -> float:
    """
    実寸世界の vmax [m/s] を返す（alpha_scaleは扱わない）
    - 非負保証
    - 学習ゾーン外は物理式へフォールバック（実寸のまま）
    - (可能なら) x_min/x_max で入力をクリップしてゾーン内で使う
    """
    if safety is None:
        safety = 0.85

    # --- 入力生成（curvatureは 1/r を徹底） ---
    r = max(float(radius_m), 1e-8)
    mu = float(mu)

    # --- 学習ゾーンへ入力クリップ（scalerが新しければ有効） ---
    if sc.get("x_min") is not None and sc.get("x_max") is not None:
        r_min, mu_min, k_min = sc["x_min"][0]
        r_max, mu_max, k_max = sc["x_max"][0]

        r  = float(np.clip(r,  r_min,  r_max))
        mu = float(np.clip(mu, mu_min, mu_max))

        k = 1.0 / max(r, 1e-8)
        k = float(np.clip(k, k_min, k_max))
    else:
        mu = max(mu, 0.0)
        k = 1.0 / max(r, 1e-8)

    x  = np.array([[r, mu, k]], np.float32)
    xn = (x - sc["x_mean"]) / sc["x_std"]

    # --- ゾーン外は物理式へフォールバック（実寸のまま） ---
    if not np.all(np.isfinite(xn)) or float(np.max(np.abs(xn))) > float(xn_abs_max):
        v = _v_phys(r, mu, g=g, safety=safety)   # ★alpha_scale を掛けない
        return max(v, 0.0)

    # --- NN 推論 ---
    with torch.no_grad():
        y_base_n = m(torch.from_numpy(xn)).numpy()

    v_base = float(y_base_n[0, 0]) * float(sc["y_std"]) + float(sc["y_mean"])
    v_base = max(v_base, 0.0)

    # --- 上限キャップ（学習ゾーンに基づく安全キャップ） ---
    if sc.get("x_max") is not None:
        r_max  = float(sc["x_max"][0, 0])
        mu_max = float(sc["x_max"][0, 1])
        v_base_cap = math.sqrt(max(mu_max, 0.0) * max(r_max, 1e-8)) * 1.2
        v_base = min(v_base, v_base_cap)

    # ★実寸m/sで返す（g,safetyはここで掛ける）
    v = v_base * (float(g) ** 0.5) * float(safety)
    return max(v, 0.0)
#def vmax_predict_factorized(m, sc, radius_m, mu, g=9.80665, safety=None):
#    # safety 未指定なら学習時の平均値は不要（Aは外的因子として掛ける設計）
#    r = max(float(radius_m), 1e-8)
#    x = np.array([[r, float(mu), 1.0/r]], np.float32)
#    xn = (x - sc["x_mean"]) / sc["x_std"]
#    with torch.no_grad():
#        y_base_n = m(torch.from_numpy(xn)).numpy()
#    v_base = y_base_n * sc["y_std"] + sc["y_mean"]  # ≈ sqrt(mu*r)
#    if safety is None:
#        safety = 0.85  # 既定（必要なら外から渡す）
#    v = float(v_base[0,0]) * (g**0.5) * float(safety)
#    return v


# 既存の import/MLP/DEF_* はそのまま

class VmaxFactorized:
    """
    使い方:
        vmax = VmaxFactorized(model_path=..., scaler_path=..., g=9.80665, safety=0.85, r_clip=1000.0)
        v = vmax(radius_m=35.0, mu=0.8)                # 単発
        v = vmax.from_kappa(kappa=0.05, mu=0.8)        # κ から
        vs = vmax.batch_radius([30,40],[0.8,0.6])      # バッチ
    """
    def __init__(self, model_path=DEF_MODEL_F, scaler_path=DEF_SCALER_F,
                 g: float = 9.80665, safety: float = 0.85,
                 #vmaxモデルの符号バグ    ★追加
#Vmaxが低すぎる問題                 
                 alpha_scale: float = 1.0,
#                 alpha_scale: float = 5.0,
                 r_clip: float = 1000.0, device: str = "cpu"):
        self.model, self.sc = load_vmax_regressor_factorized(model_path, scaler_path)
        self.g = float(g)
        self.safety = float(safety)
        #vmaxモデルの符号バグ    ★追加
        self.alpha_scale = float(alpha_scale)
        self.r_clip = float(r_clip)
        self.device = device
        # デバイスへ
        try:
            self.model.to(device)
        except Exception:
            pass
        self.model.eval()

    def _prep_r(self, r: float) -> float:
        # 直線/極小曲率対策のクリップ
        if r is None:
            return self.r_clip
        r = float(r)
        if r <= 1e-8:
            return self.r_clip
        return min(r, self.r_clip)

#vmaxモデルの符号バグ   
    @torch.no_grad()
    def __call__(self, radius_m: float, mu: float) -> float:
        # sim → real（箱庭の半径を実寸側へ拡大）
        r_real = self._prep_r(float(radius_m) * self.alpha_scale)

        # 実寸の vmax [m/s]
        v_real = vmax_predict_factorized(
            self.model, self.sc, r_real, mu,
            g=self.g, safety=self.safety
        )
        # real → sim（箱庭の速度へ戻す）
        return float(v_real) / self.alpha_scale
#    @torch.no_grad()
#    def __call__(self, radius_m: float, mu: float) -> float:
#        r = self._prep_r(radius_m)
#        return vmax_predict_factorized(
#            self.model, self.sc, r, mu,
#            g=self.g, safety=self.safety,
#            )

    @torch.no_grad()
    def from_kappa(self, kappa: float, mu: float) -> float:
        k = abs(float(kappa))
        r = self.r_clip if k < 1e-8 else min(1.0 / k, self.r_clip)
        return self(r, mu)

#vmaxモデルの符号バグ    
    @torch.no_grad()
    def batch_radius(self, radii_m, mus):
        radii = np.asarray(radii_m, np.float32).reshape(-1)
        mus   = np.asarray(mus,   np.float32).reshape(-1)
        out = np.empty_like(radii, dtype=np.float32)
        for i in range(radii.size):
            out[i] = self(float(radii[i]), float(mus[i]))
        return out
#    @torch.no_grad()
#    def batch_radius(self, radii_m, mus):
#        # radii_m, mus: list/ndarray 1D → np.float32 で返す
#        radii = np.asarray(radii_m, np.float32)
#        mus   = np.asarray(mus,   np.float32)
#        assert radii.shape == mus.shape
#        # クリップ
#        radii = np.where(radii <= 1e-8, self.r_clip, np.minimum(radii, self.r_clip)).astype(np.float32)
#
#        x = np.stack([radii, mus, 1.0 / np.maximum(radii, 1e-8)], axis=1).astype(np.float32)
#        xn = (x - self.sc["x_mean"]) / self.sc["x_std"]
#        y_base_n = self.model(torch.from_numpy(xn)).cpu().numpy()
#        v_base = y_base_n * self.sc["y_std"] + self.sc["y_mean"]  # ≈ sqrt(mu*r)
#        v = v_base[:, 0] * (self.g ** 0.5) * self.safety
#        return v.astype(np.float32)

    @torch.no_grad()
    def batch_kappa(self, kappas, mus):
        kappas = np.asarray(kappas, np.float32)
        radii  = np.where(np.abs(kappas) < 1e-8, self.r_clip, 1.0 / np.maximum(np.abs(kappas), 1e-8))
        return self.batch_radius(radii, mus)
