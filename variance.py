import numpy as np
import pandas as pd

from utils import gaussian_noise, transform_nbinom


def simulate_inv_r(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulates demand following individual negative binomial distributions.

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with modified column ``SALES``
    """
#    df["inv_r"] = np.random.uniform(0.05, 0.45, len(df))

    df["inv_r"] = 0.25
    for group_col in ["PG_ID_1", "PG_ID_2", "PG_ID_3", "P_ID", "L_ID"]:
        df["inv_r"] = df.groupby(group_col, group_keys=False)["inv_r"].apply(gaussian_noise, 0.05)

    df.loc[df["PROMOTION_TYPE"] != 0, "inv_r"] += 0.1

    df.loc[df["DATE"].dt.dayofweek == 0, "inv_r"] += 0.03
    df.loc[df["DATE"].dt.dayofweek == 1, "inv_r"] -= 0.02
    df.loc[df["DATE"].dt.dayofweek == 2, "inv_r"] += 0.01
    df.loc[df["DATE"].dt.dayofweek == 3, "inv_r"] += 0.01
    df.loc[df["DATE"].dt.dayofweek == 4, "inv_r"] -= 0.02
    df.loc[df["DATE"].dt.dayofweek == 5, "inv_r"] -= 0.03
    df.loc[df["DATE"].dt.dayofweek == 6, "inv_r"] += 0.02

    df["inv_r"].clip(lower=0.0, upper=1.0, inplace=True)

    df["variance"] = df["LAMBDA"] + df["LAMBDA"] * df["LAMBDA"] * df["inv_r"]
    df['n'], df['p'] = transform_nbinom(df["LAMBDA"], df["variance"])

    # df["SALES"] = np.random.poisson(df["LAMBDA"])
    df["SALES"] = np.random.negative_binomial(df['n'], df['p'])

    del df["inv_r"]
    del df["n"]
    del df["p"]
    del df["variance"]

    return df
