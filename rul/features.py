"""Rolling-window degradation features, computed per unit and causally (only the past `w` cycles)."""
import numpy as np
import pandas as pd

from .data import SENSORS


def build_features(df, windows=(5, 20), drop_constant=True):
    g = df.groupby("unit")
    out = [df[["unit", "cycle"]]]
    cols = [c for c in SENSORS if not (drop_constant and df[c].std() < 1e-6)]
    out.append(df[cols].add_suffix("_raw"))
    for w in windows:
        roll = g[cols].rolling(w, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True).sort_index()
        std = roll.std().reset_index(level=0, drop=True).sort_index().fillna(0.0)
        out.append(mean.add_suffix(f"_mean{w}")); out.append(std.add_suffix(f"_std{w}"))
    # slope over the last 20 cycles (least squares on a sliding window, vectorised): the classic degradation-trend feature
    w = 20
    x = np.arange(w) - (w - 1) / 2
    sl = np.zeros((len(df), len(cols)))
    for _, idx in g.indices.items():
        v = df[cols].to_numpy()[idx]
        if len(v) >= w:
            win = np.lib.stride_tricks.sliding_window_view(v, w, axis=0)           # (T-w+1, C, w)
            sl[idx[w - 1:]] = (win * x).sum(axis=2) / (x ** 2).sum()
    out.append(pd.DataFrame(sl, columns=[f"{c}_slope20" for c in cols], index=df.index))
    X = pd.concat(out, axis=1)
    return X.drop(columns=["unit", "cycle"]), cols
