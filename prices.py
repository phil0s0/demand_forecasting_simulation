import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from utils import \
    uniform_noise, \
    exponential_noise, \
    gaussian_noise


def simulate_normal_price(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulates current prices.

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with filled NORMAL_PRICE column.
    """
    mean_price = 10.0
    df["NORMAL_PRICE"] = mean_price

    df["NORMAL_PRICE"] = df.groupby("P_ID", group_keys=False)["NORMAL_PRICE"].apply(uniform_noise, 0.0, 5.0)

    for group_col in ["PG_ID_3", "PG_ID_2", "PG_ID_1"]:
        df["NORMAL_PRICE"] = df.groupby(group_col, group_keys=False)["NORMAL_PRICE"].apply(exponential_noise, 1.2)

    df["NORMAL_PRICE"] = (
        df["NORMAL_PRICE"].round(decimals=2)
    )
    df = df.loc[(df["NORMAL_PRICE"] >= 0.19) & (df["NORMAL_PRICE"] < 100.)]

    return df


def simulate_promotions(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulate promotion columns and apply promotional effects.

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with added ``PROMOTION_TYPE`` and cannibalization effect applied
    """
    # promotions (and price reductions) always valid for full week
    df["WEEK_OF_YEAR"] = df["DATE"].dt.isocalendar().week.astype(np.int16)

    df_promos = df[['P_ID', 'PG_ID_3', 'WEEK_OF_YEAR']].value_counts().reset_index()[['P_ID', 'PG_ID_3', 'WEEK_OF_YEAR']]

    # promotion confounding
    df_promos["promotion_prob"] = (df_promos['WEEK_OF_YEAR'] - df_promos['WEEK_OF_YEAR'].min()) / \
                                  (df_promos['WEEK_OF_YEAR'].max() - df_promos['WEEK_OF_YEAR'].min())
    df_promos["promotion_prob"] += (df_promos['PG_ID_3'] - df_promos['PG_ID_3'].min()) / \
                                   (df_promos['PG_ID_3'].max() - df_promos['PG_ID_3'].min())
    df_promos["promotion_prob"] /= 10.

    # two different promotion types
    df_promos["PROMOTION_TYPE"] = (np.random.rand(len(df_promos)) < df_promos["promotion_prob"]).astype(np.int16)
    df_promos.loc[
        (df_promos["PROMOTION_TYPE"] == 1) &
        (np.random.rand(len(df_promos)) < 0.2),
        "PROMOTION_TYPE"
    ] = 2

    # dedicated promotion effect only for promotion type 2
    df_promos["PROMO_FACTOR"] = 0.0
    df_promos.loc[df_promos["PROMOTION_TYPE"] == 2, "PROMO_FACTOR"] += 0.6
    df_promos.loc[df_promos["PROMOTION_TYPE"] == 2, "PROMO_FACTOR"] = \
        df_promos.loc[df_promos["PROMOTION_TYPE"] == 2].groupby("P_ID", group_keys=False)["PROMO_FACTOR"]\
            .apply(gaussian_noise, 0.1)

    del df_promos['PG_ID_3']
    del df_promos['promotion_prob']
    df = df.merge(df_promos, on=["P_ID", "WEEK_OF_YEAR"], how="left")

    # different effects for different days in promotion
    df["WEEKDAY"] = df["DATE"].dt.dayofweek.astype(np.int16)
    df.loc[(df["PROMOTION_TYPE"] == 2) & (df["WEEKDAY"] == 0), "PROMO_FACTOR"] += 0.2
    df.loc[(df["PROMOTION_TYPE"] == 2) & (df["WEEKDAY"] == 1), "PROMO_FACTOR"] += 0.1
    df.loc[(df["PROMOTION_TYPE"] == 2) & (df["WEEKDAY"] == 2), "PROMO_FACTOR"] += 0.1
    df.loc[(df["PROMOTION_TYPE"] == 2) & (df["WEEKDAY"] == 5), "PROMO_FACTOR"] += 0.3
    df.loc[(df["PROMOTION_TYPE"] == 2) & (df["WEEKDAY"] == 6), "PROMO_FACTOR"] += 0.2
    df["LOG_LAMBDA"] += df["PROMO_FACTOR"]

    df.loc[df["WEEKDAY"] == 0, "ELASTICITY"] += 0.2
    df.loc[df["WEEKDAY"] == 1, "ELASTICITY"] += 0.1
    df.loc[df["WEEKDAY"] == 2, "ELASTICITY"] += 0.1
    df.loc[df["WEEKDAY"] == 5, "ELASTICITY"] += 0.3
    df.loc[df["WEEKDAY"] == 6, "ELASTICITY"] += 0.2

    # cannibalization effect
    unique_p_id = df["P_ID"].unique()
    random_p_id = random.choices(unique_p_id, k=len(unique_p_id) // 10)
    print("cannibalizing products: ", random_p_id)
    for canni_prod in random_p_id:
        df_canni = df.loc[
            (df["PROMOTION_TYPE"] > 0) &
            (df["P_ID"] == canni_prod)
            ][["DATE", "PG_ID_3"]]
        df_canni.drop_duplicates(inplace=True)
        df_canni.reset_index(drop=True, inplace=True)
        df_canni["cannibalization"] = 1
        df = df.merge(df_canni, on=["DATE", "PG_ID_3"], how="left")
        df["cannibalization"].fillna(0, inplace=True)
        df.loc[
            (df["cannibalization"] == 1) &
            (df["P_ID"] != canni_prod),
            "LOG_LAMBDA"
        ] -= 0.8
        del df["cannibalization"]
    # df["cannibalizing"] = df["P_ID"].isin(random_p_id).astype(int)

    df.loc[df["PROMOTION_TYPE"]>0, "WEEK_OF_YEAR"].hist()
    plt.savefig("promo_season_confounding.pdf")
    plt.clf()

    del df["PROMO_FACTOR"]
    del df["WEEKDAY"]
    del df["WEEK_OF_YEAR"]

    return df


def simulate_reduced_prices(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulate `SALES_PRICE` assuming that promotion flag `PROMOTION` and the `NORMAL_PRICE`
    are already available.

    Parameters
    ----------
    df
        dataframe with NORMAL_PRICE column

    Returns
    -------
       dataframe with filled SALES_PRICE column
    """
    mean_discount_price_factor = 0.7
    unique_pids, pid_idx = np.unique(
        df["P_ID"].values, return_inverse=True
    )
    noise = pd.Series(np.random.uniform(0.7, 1.2, len(unique_pids)))[pid_idx]
    df["SALES_PRICE"] = df["NORMAL_PRICE"]
    df["SALES_PRICE"] = np.where(
        df["PROMOTION_TYPE"],
        df["SALES_PRICE"].mul(noise.values)
        * mean_discount_price_factor,
        df["SALES_PRICE"],
    )

    return df


def simulate_price_model(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulates an exponential price model and applies it to the demand.

    Parameters
    ----------
    df
        dataframe with exponential price-demand elasticity applied

    Returns
    -------
       dataframe with applied price model.
    """
    price_ratio = df["SALES_PRICE"] / df["NORMAL_PRICE"]
    df["LOG_LAMBDA"] += (1.0 - price_ratio) * df["ELASTICITY"]

    return df
