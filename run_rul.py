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

