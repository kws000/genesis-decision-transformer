# vehicle_control_dt/geo_utils.py
import numpy as np

def curvature_from_wps(waypoints, idx_now, direc, H):
    wps = np.asarray(waypoints, dtype=np.float64)
    n = len(wps)
    assert H >= 1 and n >= 3
    def wp(i): return wps[i % n][:2].astype(np.float64)
    step = 1 if direc >= 0 else -1
    kappas, ds = [], []
    for k in range(H):
        i0 = idx_now + k*step
        p_1, p0, p1 = wp(i0-1), wp(i0), wp(i0+1)
        v01, v12, v02 = p0-p_1, p1-p0, p1-p_1
        a, b, c = float(np.linalg.norm(v01)), float(np.linalg.norm(v12)), float(np.linalg.norm(v02))
        area2 = abs(float(v01[0]*v02[1] - v01[1]*v02[0]))  # 2D外積z
        kappa = float((area2/(a*b*c))*2.0) if a*b*c > 1e-6 else 0.0
        kappas.append(kappa)
        ds.append(b if b > 1e-6 else 0.0)
    return kappas, ds
