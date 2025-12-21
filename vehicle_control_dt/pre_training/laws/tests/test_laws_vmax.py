# pre_training/laws/tests/test_laws_vmax.py
from laws.laws_vmax_infer import load_vmax_regressor, vmax_predict

def test_quick_checks():
    m, sc = load_vmax_regressor(model_path="models/vmax_regressor.pt",
                                scaler_path="models/scaler.npz")
    a = vmax_predict(m, sc, 25, 0.6)
    b = vmax_predict(m, sc, 100, 0.6)
    assert b > a, "monotonicity wrt radius failed"

    c = vmax_predict(m, sc, 50, 0.0)
    assert c < 1e-3, "mu=0 should give ~0 vmax"

    x = vmax_predict(m, sc, 25, 0.8)
    y = vmax_predict(m, sc, 100, 0.8)
    ratio = y / max(x, 1e-6)
    assert 1.9 < ratio < 2.1, f"sqrt law off: ratio={ratio:.3f}"
