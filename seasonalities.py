import math
import random
from typing import List, Optional

import numpy as np
import pandas as pd

from utils import gaussian_noise


def simulate_trend(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulates linear trends over time.

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with linear trend added on the log lambda column.
    """
    unique_p_id = df["P_ID"].unique()
    random_p_id_up = random.choices(unique_p_id, k=len(unique_p_id) // 20)
    random_p_id_down = random.choices(list(set(unique_p_id) - set(random_p_id_up)), k=len(unique_p_id) // 20)
    unique_l_id = df["L_ID"].unique()
    random_l_id_up = random.choices(unique_l_id, k=len(unique_l_id) // 20)
    random_l_id_down = random.choices(list(set(unique_l_id) - set(random_l_id_up)), k=len(unique_l_id) // 20)
    print("products trending up: ", random_p_id_up)
    print("products trending down: ", random_p_id_down)
    print("locations trending up: ", random_l_id_up)
    print("locations trending down: ", random_l_id_down)

    start_date = df['DATE'].min()
    end_date = df['DATE'].max()
    df['trend'] = (df['DATE'] - start_date).dt.days

    df['trend'] = df['trend'] - (end_date - start_date).days
    df['trend_up'] = ((df['trend'] - df['trend'].min()) / (df['trend'].max() - df['trend'].min()) - 0.5) * 1.5
    df['trend_down'] = ((df['trend'].max() - df['trend']) / (df['trend'].max() - df['trend'].min()) - 0.5) * 1.5

    df.loc[df["L_ID"].isin(random_l_id_up), "LOG_LAMBDA"] += df.loc[df["L_ID"].isin(random_l_id_up), 'trend_up']
    df.loc[df["L_ID"].isin(random_l_id_down), "LOG_LAMBDA"] += df.loc[df["L_ID"].isin(random_l_id_down), 'trend_down']
    df.loc[df["P_ID"].isin(random_p_id_up), "LOG_LAMBDA"] += df.loc[df["P_ID"].isin(random_p_id_up), 'trend_up']
    df.loc[df["P_ID"].isin(random_p_id_down), "LOG_LAMBDA"] += df.loc[df["P_ID"].isin(random_p_id_down), 'trend_down']

    del df["trend"]
    del df["trend_up"]
    del df["trend_down"]

    return df


def simulate_weekday_profile(
    df: pd.DataFrame,
    weekday_profile: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Simulates weekday profile.

    Parameters
    ----------
    df
        dataframe
    weekday_profile
        List containing the weekday profile that the demand should follow.

    Returns
    -------
        dataframe with weekday profile effect added on the log lambda column.
    """
    demand_group_cols = [
        "PG_ID_1",
        "PG_ID_2",
        "PG_ID_3",
        "P_ID",
        "L_ID",
    ]

    # global weekday profile
    if weekday_profile is None:
        weekday_profile = np.log([1.0, 0.7, 0.8, 0.9, 1.3, 1.5, 1.1])
    date_dict = dict(zip(range(len(weekday_profile)), weekday_profile))
    df["LOG_LAMBDA"] += df["DATE"].dt.weekday.replace(date_dict)

    # individual weekday profiles
    df["WEEKDAY"] = df["DATE"].dt.weekday
    for group_col in demand_group_cols:
        df["LOG_LAMBDA"] = df.groupby(["WEEKDAY", group_col], group_keys=False)["LOG_LAMBDA"].apply(gaussian_noise, 0.1)

    for group_col in ["PG_ID_1", "PG_ID_2", "PG_ID_3", "P_ID"]:
        df["LOG_LAMBDA"] = df.groupby(["WEEKDAY", "L_ID", group_col], group_keys=False)["LOG_LAMBDA"].apply(gaussian_noise, 0.1)

    del df["WEEKDAY"]

    return df


def simulate_monthly_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulates monthly profile.

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with monthly profile effect for sampled locations and product groups 3 added on the log lambda column.
    """
    unique_pg3 = df["PG_ID_3"].unique()
    random_pg_3 = random.choices(unique_pg3, k=len(unique_pg3) // 2)
    unique_l_id = df["L_ID"].unique()
    random_l_id = random.choices(unique_l_id, k=len(unique_l_id) // 2)

    df["dayofmonth"] = df["DATE"].dt.day
    monthly_profile = np.log([1.5, 1.3, 1.1])
    date_dict = dict(zip(range(1, len(monthly_profile) + 1), monthly_profile))
    mask = (df["dayofmonth"] <= 3)

    df.loc[mask, "LOG_LAMBDA"] = (
        df.loc[mask, "LOG_LAMBDA"]
        .add(df.loc[mask, "dayofmonth"].replace(date_dict) +
             np.random.normal(scale=0.1, size=len(df.loc[mask])))
        .where(
            df.loc[mask, "PG_ID_3"].isin(random_pg_3) & df.loc[mask, "L_ID"].isin(random_l_id),
            df.loc[mask, "LOG_LAMBDA"]
        )
    )

    del df["dayofmonth"]

    return df


def simulate_yearly_seasonality(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulates yearly seasonality in lambda_col for a selection of product/locations.
    The products are chosen as following: First, random pg3s are chosen, then seasonality is applied to all products in
    those pgs (this guarantees that we have pg3 X seasonality interaction)

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with added seasonality effect
    """
    unique_pg3 = df["PG_ID_3"].unique()
    random_pg_3 = random.choices(unique_pg3, k=len(unique_pg3) // 2)
    df["DAY_OF_YEAR"] = df["DATE"].dt.dayofyear.astype(np.int16) - 1
    df["yearly_seas"] = np.sin(2 * math.pi * df["DAY_OF_YEAR"] / 365)
    df["half_yearly_seas"] = np.sin(2 * math.pi * df["DAY_OF_YEAR"] / (365 // 2))
    # get all products in the pg3
    unique_prods = []
    for pg3 in random_pg_3:
        unique_prods += list(
            df.loc[df["PG_ID_3"] == pg3, "P_ID"].unique()
        )
    # extract present locations
    locations = list(
        df.loc[df["P_ID"].isin(unique_prods)]["L_ID"].unique()
    )
    # pick random locations
    random_locations = random.choices(locations, k=len(locations) // 2)

    df["LOG_LAMBDA"] = (
        df["LOG_LAMBDA"]
        .add(df["yearly_seas"] * np.random.exponential(0.5, len(df["yearly_seas"])).clip(max=1.5))
        .where(
            df["P_ID"].isin(unique_prods[: len(unique_prods) // 2])
            & df["L_ID"].isin(random_locations),
            df["LOG_LAMBDA"]
        )
    )
    df["LOG_LAMBDA"] = (
        df["LOG_LAMBDA"]
        .add(df["half_yearly_seas"] * np.random.exponential(0.5, len(df["half_yearly_seas"])).clip(max=1.5))
        .where(
            df["P_ID"].isin(unique_prods[len(unique_prods) // 2 :])
            & df["L_ID"].isin(random_locations),
            df["LOG_LAMBDA"]
        )
    )

    del df["yearly_seas"]
    del df["half_yearly_seas"]
    del df["DAY_OF_YEAR"]

    return df


def simulate_school_holidays(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulates different weekly profile for school holidays.

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with alterd weekly profile in school holidays for sampled locations and product groups 3 added on the log lambda column.
    """
    unique_pg3 = df["PG_ID_3"].unique()
    random_pg_3 = random.choices(unique_pg3, k=len(unique_pg3) // 2)
    unique_l_id = df["L_ID"].unique()
    random_l_id = random.choices(unique_l_id, k=len(unique_l_id) // 3)

    i = pd.date_range('2020-06-08', '2020-07-17')
    j = pd.date_range('2021-06-28', '2021-08-06')
    k = pd.date_range('2022-07-04', '2022-08-12')
    i = i.append(j).append(k)
    school_holidays = pd.DataFrame({'SCHOOL_HOLIDAY': 1}, index=i)
    school_holidays.reset_index(level=0, inplace=True)
    school_holidays.rename(columns={"index": "DATE"}, inplace=True)
    df = df.merge(school_holidays, on="DATE", how="left")
    #df["SCHOOL_HOLIDAY"].fillna(0, inplace=True)
    df["SCHOOL_HOLIDAY"] = df["SCHOOL_HOLIDAY"].fillna(0)

    weekday_profile = np.log([1.5, 1.4, 1.3, 1.0, 0.8, 0.7, 0.7])
    date_dict = dict(zip(range(len(weekday_profile)), weekday_profile))
    df["LOG_LAMBDA"] += df["DATE"].dt.weekday.replace(date_dict)
    df["LOG_LAMBDA"] = (
        df["LOG_LAMBDA"]
        .add(df["DATE"].dt.weekday.replace(date_dict))
        .where(
            df["PG_ID_3"].isin(random_pg_3)
            & df["L_ID"].isin(random_l_id),
            df["LOG_LAMBDA"]
        )
    )

    return df
