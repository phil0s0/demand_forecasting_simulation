from typing import Optional, Tuple

import numpy as np
import pandas as pd


def gaussian_noise(
        x: pd.Series,
        sigma: float,
        mu: Optional[float] = 0.,
) -> pd.Series:
    x += np.random.normal(loc=mu, scale=sigma)
    return x


def uniform_noise(
        x: pd.Series,
        lower: float,
        higher: float,
) -> pd.Series:
    x += np.random.uniform(lower, higher, len(x))
    return x


def exponential_noise(
        x: pd.Series,
        beta: float,
) -> pd.Series:
    x *= np.random.exponential(beta, len(x))
    return x

def transform_nbinom(
        mean: float,
        var: float,
) -> Tuple[float, float]:
    p = np.minimum(np.where(var > 0, mean / var, 1 - 1e-8), 1 - 1e-8)
    n = np.where(var > 0, mean * p / (1 - p), 1)
    return n, p
