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

