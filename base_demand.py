import numpy as np
import pandas as pd

from utils import gaussian_noise


def simulate_products(
    n_products: int,
    p_id_shift: int,
) -> pd.DataFrame:
    """
    Simulates products (product IDs).

    Parameters
    ----------
    n_products
        number of products to simulate
    p_id_shift
        lowest value for product IDs

    Returns
    -------
        dataframe with added product ID column
    """
    p_ids = np.arange(p_id_shift, n_products + p_id_shift)
    df = pd.DataFrame(
        {
            "P_ID": p_ids,
        }
    )

    return df


def simulate_product_groups(
        df: pd.DataFrame,
        n_pg3: int,
        pg_id_shift: int,
) -> pd.DataFrame:
    """
    Simulates 3 product groups.

    Parameters
    ----------
    df
        dataframe
    n_pg3
        Number of level-3 product groups
    pg_id_shift
        lowest value for product group IDs

    Returns
    -------
        dataframe with product group columns
    """
    df["PG_ID_3"] = df["P_ID"].values % n_pg3 + pg_id_shift
    df["PG_ID_2"] = df["PG_ID_3"] // 8 + pg_id_shift
    df["PG_ID_1"] = df["PG_ID_2"] // 3 + pg_id_shift

    return df


def simulate_locations(
        df: pd.DataFrame,
        n_locations: int,
        l_id_shift: int,
) -> pd.DataFrame:
    """
    Adds location-specific columns.

    Parameters
    ----------
    df
        input dataframe
    n_locations
        number of locations
    l_id_shift
        lowest value for location IDs

    Returns
    -------
       extended dataframe with added location columns
    """
    l_ids = np.arange(l_id_shift, n_locations + l_id_shift)
    df_loc = pd.DataFrame(
        {
            "L_ID": l_ids,
            "SALES_AREA": np.random.uniform(1000,10000, size=len(l_ids)),
            "INTERNAL:TMP": 0,
        }
    )
    df["INTERNAL:TMP"] = 0
    df = df.merge(df_loc, on="INTERNAL:TMP", how="outer")
    del df["INTERNAL:TMP"]

    return df


def simulate_dates(
    df: pd.DataFrame,
    date_from: str,
    date_upto: str,
) -> pd.DataFrame:
    """
    Simulates dates for each product-location combination.

    Parameters
    ----------
    df
        dataframe
    date_from
        date from
    date_upto
        date upto

    Returns
    -------
        dataframe with added columns ``DATE_COL``
    """
    df_dates = pd.DataFrame(
        {
            "DATE": pd.date_range(start=date_from, end=date_upto),
            "INTERNAL:TMP": 0,
        }
    )

    df["INTERNAL:TMP"] = 0
    df = df.merge(df_dates, on="INTERNAL:TMP", how="outer")
    del df["INTERNAL:TMP"]

    return df


def simulate_basic_demand(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulates basic demand by applying a random noise to different products and locations.

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with modified columns ``LOG_LAMBDA`` and ``ELASTICITY``
    """
    df["LOG_LAMBDA"] = 0.5
    df["LOG_LAMBDA"] = df.groupby("PG_ID_1", group_keys=False)["LOG_LAMBDA"].apply(gaussian_noise, 0.5)
    df["LOG_LAMBDA"] = df.groupby("L_ID", group_keys=False)["LOG_LAMBDA"].apply(gaussian_noise, 0.5)
    df["LOG_LAMBDA"] = df.groupby("PG_ID_2", group_keys=False)["LOG_LAMBDA"].apply(gaussian_noise, 0.3)
    df["LOG_LAMBDA"] = df.groupby("PG_ID_3", group_keys=False)["LOG_LAMBDA"].apply(gaussian_noise, 0.3)
    df["LOG_LAMBDA"] = df.groupby("P_ID", group_keys=False)["LOG_LAMBDA"].apply(gaussian_noise, 0.3)
    df["LOG_LAMBDA"] = df.groupby(["L_ID", "P_ID"], group_keys=False)["LOG_LAMBDA"].apply(gaussian_noise, 0.1)

    # normal-price effect on demand
    median_price = df["NORMAL_PRICE"].median()
    df.loc[df["NORMAL_PRICE"] < median_price, "LOG_LAMBDA"] = \
        df.loc[df["NORMAL_PRICE"] < median_price].groupby("P_ID", group_keys=False)["LOG_LAMBDA"]\
            .apply(gaussian_noise, 0.1, mu=0.3)
    df.loc[df["NORMAL_PRICE"] >= median_price, "LOG_LAMBDA"] = \
        df.loc[df["NORMAL_PRICE"] >= median_price].groupby("P_ID", group_keys=False)["LOG_LAMBDA"]\
            .apply(gaussian_noise, 0.1, mu=-0.3)

    # sales-area effect on demand
    mean_sales_area = df["SALES_AREA"].mean()
    df.loc[df["SALES_AREA"] > mean_sales_area, "LOG_LAMBDA"] = \
        df.loc[df["SALES_AREA"] > mean_sales_area].groupby("L_ID", group_keys=False)["LOG_LAMBDA"]\
            .apply(gaussian_noise, 0.1, mu=0.3)
    df.loc[df["SALES_AREA"] <= mean_sales_area, "LOG_LAMBDA"] = \
        df.loc[df["SALES_AREA"] <= mean_sales_area].groupby("L_ID", group_keys=False)["LOG_LAMBDA"]\
            .apply(gaussian_noise, 0.1, mu=-0.3)

    elasticity_group_cols = [
        "PG_ID_1",
        "PG_ID_2",
        "PG_ID_3",
        "P_ID",
        "L_ID",
    ]
    df["ELASTICITY"] = np.log(1.5)
    for group_col in elasticity_group_cols:
        df["ELASTICITY"] = df.groupby(group_col, group_keys=False)["ELASTICITY"].apply(gaussian_noise, 0.3)
    df["ELASTICITY"] = np.exp(df["ELASTICITY"])
    df["ELASTICITY"].clip(lower=0.0, upper=3.0, inplace=True)

    return df
