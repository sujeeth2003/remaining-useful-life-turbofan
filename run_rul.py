"""Predict remaining useful life; evaluate on held-out ENGINES (never random rows: rows of one engine are almost identical
neighbours, so a random row split leaks and looks far better than reality).

    python run_rul.py                                   # synthetic fleet
    python run_rul.py --cmapss path/to/train_FD001.txt  # real NASA C-MAPSS data

Reports RMSE over ALL cycles and, more usefully, over each engine's LAST observed cycle (the moment a maintenance decision
is actually made), plus the asymmetric NASA score.
"""
import argparse

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from rul.data import load_cmapss, synthetic_fleet
from rul.features import build_features
from rul.metrics import nasa_score, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmapss", help="path to a C-MAPSS train_FDxxx.txt file")
    ap.add_argument("--units", type=int, default=100)
    ap.add_argument("--folds", type=int, default=3)
    a = ap.parse_args()

    df = load_cmapss(a.cmapss) if a.cmapss else synthetic_fleet(a.units)
    src = a.cmapss or f"synthetic fleet ({a.units} engines)"
    X, cols = build_features(df)
    y, groups = df["rul_clipped"].to_numpy(), df["unit"].to_numpy()
    print(f"data: {src}: {len(df)} rows, {df['unit'].nunique()} engines, {X.shape[1]} features from {len(cols)} sensors\n")

    models = {
        "baseline: predict train mean": None,
        "random forest": lambda: RandomForestRegressor(n_estimators=80, min_samples_leaf=5, max_features=0.3, random_state=0),
        "gradient boosting": lambda: GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.7, max_features=0.3, random_state=0),
    }
    res = {k: {"all": [], "last": [], "score": []} for k in models}
    for tr, te in GroupKFold(n_splits=a.folds).split(X, y, groups):
        last_idx = df.iloc[te].groupby("unit")["cycle"].idxmax().to_numpy()
        last_pos = np.flatnonzero(np.isin(te, last_idx))
        for name, make in models.items():
            pred = np.full(len(te), y[tr].mean()) if make is None else make().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            res[name]["all"].append(rmse(y[te], pred))
            res[name]["last"].append(rmse(y[te][last_pos], pred[last_pos]))
            res[name]["score"].append(nasa_score(y[te][last_pos], pred[last_pos]))
    print(f"{'model':<30}{'RMSE (all cycles)':>19}{'RMSE (last cycle)':>19}{'NASA score (last)':>19}")
    for name, r in res.items():
        print(f"{name:<30}{np.mean(r['all']):>12.1f} +/-{np.std(r['all']):>4.1f}{np.mean(r['last']):>12.1f} +/-{np.std(r['last']):>4.1f}{np.mean(r['score']):>19.0f}")
    print(f"\n({a.folds}-fold cross-validation grouped by engine; RUL target clipped at 125 cycles)")


if __name__ == "__main__":
    main()
