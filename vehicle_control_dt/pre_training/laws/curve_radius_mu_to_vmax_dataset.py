#!/usr/bin/env python3
"""
curve_radius_mu_to_vmax_dataset.py

Purpose
-------
Generate a "laws-only" synthetic dataset that encodes the flat-road lateral
friction limit: the maximum feasible speed in a curve of radius r given road
friction coefficient mu (no banking), using the classical relation

    v_max = sqrt(mu * g * r) * safety_margin

Where
- r [m] is curve radius (r > 0)
- mu [-] is the friction coefficient (e.g., dry asphalt ~0.9, wet ~0.5)
- g [m/s^2] is gravitational acceleration (default 9.80665)
- safety_margin in (0,1] encodes conservatism for comfort/uncertainty

This script produces a CSV of i.i.d. or grid-sampled (r, mu) pairs with the
resulting v_max in both m/s and km/h, plus auxiliary columns. It is designed
as the first "座学(lecture)" dataset for constraint-driven DT pretraining.

Why this dataset?
-----------------
- It teaches the model a universal physical constraint, independent of any
  specific map: lateral acceleration <= mu * g.
- It is small, fast to generate, and numerically well-behaved.
- It is extensible (bank angle, grade, uncertainty, noise models, etc.).

Example
-------
# Uniform random 100k samples across radius [10, 500] m and mu [0.2, 1.1]
python3 curve_radius_mu_to_vmax_dataset.py \
  --mode random --n 100000 \
  --r-min 10 --r-max 500 \
  --mu-min 0.2 --mu-max 1.1 \
  --safety 0.85 \
  --out data/curve_mu_vmax_random.csv

# Log-spaced radii (dense at tight curves), grid over mu
python3 curve_radius_mu_to_vmax_dataset.py \
  --mode grid --r-points 60 --r-scale log --r-min 8 --r-max 1000 \
  --mu-points 10 --mu-min 0.3 --mu-max 1.0 \
  --safety 0.9 \
  --out data/curve_mu_vmax_grid.csv

Notes
-----
- This is a *flat-road* baseline. Banking or combined accel can be added later.
- Optionally inject tiny observation noise (e.g., sensor jitter) to help
  robustness.
- "curvature" = 1/r is included because many planners/reward terms use it.

Author: ChatGPT (for 川崎さん)
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

# ------------------------------ Physics Core ------------------------------ #

def vmax_flat_radius_mu(r: np.ndarray, mu: np.ndarray, g: float, safety: float) -> np.ndarray:
    """Compute maximum speed [m/s] on flat road from radius and friction.

    v_max = sqrt(mu * g * r) * safety

    Args:
        r: radius [m], must be > 0
        mu: friction coefficient [-], must be >= 0
        g: gravitational acceleration [m/s^2]
        safety: multiplicative margin (0 < safety <= 1)
    Returns:
        v_max [m/s]
    """
    r = np.asarray(r, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if np.any(r <= 0):
        raise ValueError("All radii must be > 0.")
    if np.any(mu < 0):
        raise ValueError("All mu must be >= 0.")
    if not (0 < safety <= 1.0):
        raise ValueError("safety must be in (0, 1].")
    return np.sqrt(mu * g * r) * safety

# ------------------------------ Sampling --------------------------------- #

@dataclass
class SamplingConfig:
    mode: Literal["random", "grid"] = "random"
    n: int = 10000                  # for random mode
    r_min: float = 10.0
    r_max: float = 500.0
    r_points: int = 50              # for grid mode
    r_scale: Literal["linear", "log"] = "linear"
    mu_min: float = 0.2
    mu_max: float = 1.1
    mu_points: int = 20             # for grid mode
    seed: int = 42
    jitter_std_r: float = 0.0       # optional obs noise on r [m]
    jitter_std_mu: float = 0.0      # optional obs noise on mu [-]


def sample_radius_mu(cfg: SamplingConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    if cfg.mode == "random":
        r = rng.uniform(cfg.r_min, cfg.r_max, size=cfg.n)
        mu = rng.uniform(cfg.mu_min, cfg.mu_max, size=cfg.n)

    elif cfg.mode == "grid":
        if cfg.r_scale == "linear":
            r_vals = np.linspace(cfg.r_min, cfg.r_max, cfg.r_points)
        elif cfg.r_scale == "log":
            if cfg.r_min <= 0:
                raise ValueError("r_min must be > 0 for log spacing.")
            r_vals = np.geomspace(cfg.r_min, cfg.r_max, cfg.r_points)
        else:
            raise ValueError("r_scale must be 'linear' or 'log'.")

        mu_vals = np.linspace(cfg.mu_min, cfg.mu_max, cfg.mu_points)
        R, MU = np.meshgrid(r_vals, mu_vals, indexing='xy')
        r = R.ravel()
        mu = MU.ravel()
    else:
        raise ValueError("mode must be 'random' or 'grid'.")

    # Optional observation jitter (kept small & clipped to maintain validity)
    if cfg.jitter_std_r > 0:
        r = np.clip(r + rng.normal(0.0, cfg.jitter_std_r, size=r.shape), 1e-6, None)
    if cfg.jitter_std_mu > 0:
        mu = np.clip(mu + rng.normal(0.0, cfg.jitter_std_mu, size=mu.shape), 0.0, None)

    return r, mu

# ------------------------------ Dataset Build ----------------------------- #

@dataclass
class BuildConfig:
    g: float = 9.80665
    safety: float = 0.9


def build_dataframe(r: np.ndarray, mu: np.ndarray, build: BuildConfig) -> pd.DataFrame:
    v_mps = vmax_flat_radius_mu(r, mu, build.g, build.safety)
    v_kph = v_mps * 3.6
    curvature = 1.0 / r

    df = pd.DataFrame({
        'radius_m': r,
        'mu': mu,
        'g_mps2': build.g,
        'safety_margin': build.safety,
        'vmax_mps': v_mps,
        'vmax_kph': v_kph,
        'curvature_1pm': curvature,  # 1/m
    })

    # Sorted for deterministic diffs (first by mu, then by radius)
    df.sort_values(['mu', 'radius_m'], inplace=True, kind='mergesort')
    df.reset_index(drop=True, inplace=True)
    return df

# ------------------------------ CLI -------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate r,mu -> v_max dataset (flat road)")

    # Sampling
    p.add_argument('--mode', choices=['random', 'grid'], default='random')
    p.add_argument('--n', type=int, default=10000, help='samples for random mode')

    p.add_argument('--r-min', type=float, default=10.0)
    p.add_argument('--r-max', type=float, default=500.0)
    p.add_argument('--r-points', type=int, default=50, help='grid only')
    p.add_argument('--r-scale', choices=['linear', 'log'], default='linear')

    p.add_argument('--mu-min', type=float, default=0.2)
    p.add_argument('--mu-max', type=float, default=1.1)
    p.add_argument('--mu-points', type=int, default=20, help='grid only')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--jitter-std-r', type=float, default=0.0, help='obs noise on r [m]')
    p.add_argument('--jitter-std-mu', type=float, default=0.0, help='obs noise on mu [-]')

    # Physics
    p.add_argument('--g', type=float, default=9.80665)
    p.add_argument('--safety', type=float, default=0.9)

    # I/O
    p.add_argument('--out', type=str, default='curve_mu_vmax.csv')
    p.add_argument('--no-header', action='store_true', help='omit CSV header')




    return p.parse_args()


def main():
    args = parse_args()

    samp = SamplingConfig(
        mode=args.mode,
        n=args.n,
        r_min=args.r_min,
        r_max=args.r_max,
        r_points=args.r_points,
        r_scale=args.r_scale,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        mu_points=args.mu_points,
        seed=args.seed,
        jitter_std_r=args.jitter_std_r,
        jitter_std_mu=args.jitter_std_mu,
    )

    build = BuildConfig(g=args.g, safety=args.safety)

    r, mu = sample_radius_mu(samp)
    df = build_dataframe(r, mu, build)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, header=not args.no_header)

    # Human-friendly stdout summary
    print("Saved:", out_path)
    print("Rows:", len(df))
    print(df.describe(include='all'))


if __name__ == '__main__':
    main()
