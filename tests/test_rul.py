import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rul.data import add_rul, synthetic_fleet  # noqa: E402
from rul.features import build_features  # noqa: E402
from rul.metrics import nasa_score, rmse  # noqa: E402


class RULTests(unittest.TestCase):
    def test_rul_counts_down_to_zero_and_is_clipped(self):
        df = synthetic_fleet(5, seed=1)
        for _, g in df.groupby("unit"):
            self.assertEqual(g["rul"].iloc[-1], 0)
            self.assertTrue((np.diff(g["rul"]) == -1).all())
        self.assertLessEqual(df["rul_clipped"].max(), 125)

    def test_features_are_causal(self):
        df = synthetic_fleet(3, seed=2)
        X1, _ = build_features(df)
        df2 = df.copy()
        u = df2["unit"] == 1
        late = u & (df2["cycle"] > 100)
        df2.loc[late, [c for c in df2.columns if c.startswith("x")]] += 50      # change unit 1's future
        X2, _ = build_features(df2)
        early = (df["unit"] == 1) & (df["cycle"] <= 100)
        np.testing.assert_allclose(X1[early].to_numpy(), X2[early].to_numpy())

