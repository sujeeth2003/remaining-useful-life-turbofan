import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def nasa_score(y, p):
    """Asymmetric PHM08 score: late predictions (overestimating RUL) are penalised more than early ones,
    because a missed failure costs more than an early maintenance. Lower is better."""
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))
