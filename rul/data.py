"""Data loading for run-to-failure engine data.

Two sources, same format (columns: unit, cycle, 3 operating settings, 21 sensors):
  * NASA C-MAPSS turbofan, e.g. train_FD001.txt  (download from the NASA Prognostics Data Repository; not redistributed here)
  * a synthetic generator with the same shape and qualitative behaviour, so the pipeline runs anywhere
"""
import numpy as np
import pandas as pd

SETTINGS = ["s1", "s2", "s3"]
SENSORS = [f"x{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS


def load_cmapss(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return add_rul(df)


def add_rul(df, clip=125):
    """RUL = cycles remaining until the unit's last recorded cycle; clipped (early life is not informative)."""
    last = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = last - df["cycle"]
    df["rul_clipped"] = df["rul"].clip(upper=clip)
    return df


def synthetic_fleet(n_units=100, seed=0, informative=8):
    """Each engine degrades exponentially towards failure. `informative` sensors carry the degradation signal with
    per-unit noise and a random per-unit offset (manufacturing spread); the rest are noise or constant, like C-MAPSS."""
    rng = np.random.default_rng(seed)
    rows = []
    weights = rng.uniform(0.4, 1.0, informative) * rng.choice([-1, 1], informative)
    for u in range(1, n_units + 1):
        life = int(rng.integers(150, 360))
        offs = rng.normal(0, 0.6, len(SENSORS))
        onset = int(life * rng.uniform(0.25, 0.5))                 # degradation starts part-way through life
        for c in range(1, life + 1):
            d = 0.0 if c < onset else ((c - onset) / (life - onset)) ** 2.0     # 0 -> 1 at failure
            s = rng.normal(0, 0.35, len(SENSORS)) + offs
            s[:informative] += weights * 4.0 * d
            settings = rng.normal(0, 0.02, 3)
            rows.append([u, c, *settings, *s])
    return add_rul(pd.DataFrame(rows, columns=COLUMNS))
