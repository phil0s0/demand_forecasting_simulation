import numpy as np
import pandas as pd


def balance(
    s: pd.Series,
    gain: float = 0.5,
) -> pd.Series:
    """
    calulate rough balansed log lambda. 
    because coupled product demand seems to be similar.

    Parameters
    ----------
    s
        series
    gain
        demand increase ratio to lower demand products in a couple

    Returns
    -------
        series with rough balansed log lambda
    """

    _max = s.max()
    s = s + (_max - s) * gain
    
    return s


def simulate_coupling_demand(
    df: pd.DataFrame,
    n_couples: int,
    max_products: int = 3,
) -> pd.DataFrame:
    """
    Simulates coupling demand.
    In many cases, Bread and Jam are usually bought same time.

    Parameters
    ----------
    df
        dataframe
    n_couples
        number of couples to simulate
    max_products
        max number of products included in one-couple

    Returns
    -------
        dataframe with coupling demand effect added on the log lambda column.
    """

    p_id = df['P_ID'].unique()
    df['C_ID'] = -1 # not couple

    # couple
    for c_id in range(n_couples):
        prods = np.random.randint(low=2, high=max_products+1)
        couple = np.random.choice(p_id, size=prods, replace=False)
        mask = df['P_ID'].isin(couple)
        df.loc[mask, 'C_ID'] = c_id
        p_id = list(set(p_id) - set(couple)) # not allow overlapping

    # add coupling demand effect for couple
    mask = df['C_ID'] != -1
    df.loc[mask, 'LOG_LAMBDA'] = df.loc[mask].groupby(["C_ID", "L_ID"],
                                   group_keys=False)['LOG_LAMBDA'].apply(
                                   balance, np.random.uniform(0.0, 0.5))

    df.drop(columns=['C_ID'], inplace=True)

    return df