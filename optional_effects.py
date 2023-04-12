import random

import numpy as np
import pandas as pd
from datetime import timedelta


def _balance(
    s: pd.Series,
    gain: float = 0.,
) -> pd.Series:
    """
    calulates roughly balanced log lambda.

    Parameters
    ----------
    s
        series
    gain
        demand-increase ratio for lower-demand products in a couple

    Returns
    -------
        series with roughly balanced log lambda
    """

    return s + (s.max() - s) * gain


def simulate_coupled_demand(
    df: pd.DataFrame,
    n_couples: int,
    max_products: int = 3,
) -> pd.DataFrame:
    """
    Simulates coupled-demand effects.
    In many cases, things like bread and jam are bought together.

    Parameters
    ----------
    df
        dataframe
    n_couples
        number of couples to simulate
    max_products
        max number of products included in one couple

    Returns
    -------
        dataframe with coupled-demand effect added on the log lambda column.
    """

    p_id = df['P_ID'].unique()
    df['C_ID'] = -1 # no couple

    # couple
    for c_id in range(n_couples):
        prods = np.random.randint(low=2, high=max_products+1)
        couple = np.random.choice(p_id, size=prods, replace=False)
        mask = df['P_ID'].isin(couple)
        df.loc[mask, 'C_ID'] = c_id
        p_id = list(set(p_id) - set(couple)) # not allow overlapping

    # add coupled-demand effect for different couples
    mask = df['C_ID'] != -1
    df.loc[mask, 'LOG_LAMBDA'] = df.loc[mask].groupby(["C_ID", "L_ID"],
                                   group_keys=False)['LOG_LAMBDA'].apply(
                                   _balance, np.random.uniform(0.0, 0.5))

    df.drop(columns=['C_ID'], inplace=True)

    return df


def drop_zeros(
        df: pd.DataFrame
) -> pd.DataFrame:
    return df.loc[df['SALES'] != 0]


def add_anomalies(
        df: pd.DataFrame,
        prob_huge = 0.,
        prob_neg = 0.
) -> pd.DataFrame:
    """
    randomly assign a small number of sales as huge or negative
    """
    n = len(df)
    anomaly_list = [1] * n
    for i in range(n):
        if random.random() < prob_huge:
            anomaly_list[i] = random.randint(1000, 10000)  # huge number
        elif random.random() < prob_neg:
            anomaly_list[i] = random.randint(-10, -1)

    df['SALES'] = df['SALES'] * anomaly_list

    return df


def restrict_pl_ranges(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    NPI and discontinuation
    """
    # products
    df_cut_dates = df.groupby(['P_ID']).DATE.max().reset_index()
    df_cut_dates['START_DATE'] = [df_cut_dates.DATE.max() - timedelta(days=random.randint(1, 5000)) for _ in
                                  range(len(df_cut_dates))]
    df_cut_dates['END_DATE'] = [df_cut_dates.DATE.max() + timedelta(days=9000) - timedelta(days=random.randint(1, 10000))
                                for _ in range(len(df_cut_dates))]
    df = df.merge(df_cut_dates[['P_ID', 'START_DATE', 'END_DATE']], on=['P_ID'], how='left')
    df = df[(df.DATE > df.START_DATE) & (df.DATE < df.END_DATE)]
    df.drop(columns=['START_DATE', 'END_DATE'], inplace=True)

    # product-location combinations
    df_cut_dates = df.groupby(['P_ID', 'L_ID']).DATE.max().reset_index()
    df_cut_dates['START_DATE'] = [df_cut_dates.DATE.max() - timedelta(days=random.randint(1, 5000)) for _ in
                                  range(len(df_cut_dates))]
    df_cut_dates['END_DATE'] = [df_cut_dates.DATE.max() + timedelta(days=9000) - timedelta(days=random.randint(1, 10000))
                                for _ in range(len(df_cut_dates))]
    df = df.merge(df_cut_dates[['P_ID', 'L_ID', 'START_DATE', 'END_DATE']], on=['P_ID', 'L_ID'], how='left')
    df = df[(df.DATE > df.START_DATE) & (df.DATE < df.END_DATE)]
    df.drop(columns=['START_DATE', 'END_DATE'], inplace=True)

    return df
