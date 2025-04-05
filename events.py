import random
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, Easter


def simulate_events(
        df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulates events following two event types by following these steps.
    1. Sample random product/ locations to apply the events effect on.
    2. Construct PandasEventCalendar from a predefined events list.
    3. Construct df_events: a data frame that contains the exact date for every event every year
    4. For every event in every year, construct date_from which is 7 days before the event's date and date_upto
    which is 3 days after the event's date
    5. We consider Christmas and Labour Day as event type1, two local holidays as event type 2 and two noise events.
    event_type_1: For this sort of event, we observe a peak before the event's date and a dip afterwards.
    event_type_2: For this sort of event, we only observe a peak before the event's date.
    event_type_3, for this event, no effect is observed.
    For each event, we multiply the demand by the observed effect.

    Parameters
    ----------
    df
        dataframe

    Returns
    -------
        dataframe with added events effect visible in the SALES column.
    """
    events_list = [
        Holiday("Christmas", month=12, day=25),
        Holiday('Easter', month=1, day=1, offset=[Easter()]),
        Holiday("Labour_Day", month=5, day=1),
        Holiday("German_Unity", month=10, day=3),
        Holiday("Other_Holiday", month=9, day=22),
        Holiday("Local_Holiday_0", month=3, day=13),
        Holiday("Local_Holiday_1", month=6, day=26),
        Holiday("Local_Holiday_2", month=5, day=13),
    ]
    # apply over 20 days: 10 days before the event and 10 days after the event.
    event_type_1 = np.log([
        1., 1., 1., 1.2, 1.4, 1.6, 1.8, 2., 2.5, 3.,
        0.0001,
        0.7, 1.3, 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.2, 1.4, 1.6, 1.8, 2., 2.5, 3.,
        0.0001,
        0.7, 1.3, 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.2, 1.4, 1.6, 1.8, 2., 2.5, 3.,
        0.0001,
        0.7, 1.3, 1., 1., 1., 1., 1., 1., 1., 1.,
    ])
    event_type_2 = np.log([
        1., 1., 1., 1.5, 2., 3.2, 3.6, 4., 3.2, 2.5,
        2.,
        0.7, 0.8, 1, 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.5, 2., 3.2, 3.6, 4., 3.2, 2.5,
        2.,
        0.7, 0.8, 1, 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.5, 2., 3.2, 3.6, 4., 3.2, 2.5,
        2.,
        0.7, 0.8, 1, 1., 1., 1., 1., 1., 1., 1.,
    ])
    christmas_type = np.log([
        1.2, 1.4, 1.6, 1.8, 2.5, 3., 3.5, 3.8, 4., 5.,
        1.8,
        0.0001, 0.0001, 2.9, 2.7, 2.5, 4.5, 2.1, 0.0001, 1.5, 1.2,
        1.2, 1.4, 1.6, 1.8, 2.5, 3., 3.5, 3.8, 4., 5.,
        1.8,
        0.0001, 0.0001, 2.9, 2.7, 2.5, 4.5, 2.1, 0.0001, 1.5, 1.2,
        1.2, 1.4, 1.6, 1.8, 2.5, 3., 3.5, 3.8, 4., 5.,
        1.8,
        0.0001, 0.0001, 2.9, 2.7, 2.5, 4.5, 2.1, 0.0001, 1.5, 1.2,
    ])
    easter_type = np.log([
        1.1, 1.3, 1.5, 1.7, 2.8, 3.5, 3.8, 5., 0.0001, 4.5,
        0.0001,
        0.0001, 1.5, 1.3, 0.8, 0.7, 0.9, 1., 1., 1., 1.,
        1.1, 1.3, 1.5, 1.7, 2.8, 3.5, 3.8, 5., 0.0001, 4.5,
        0.0001,
        0.0001, 1.5, 1.3, 0.8, 0.7, 0.9, 1., 1., 1., 1.,
        1.1, 1.3, 1.5, 1.7, 2.8, 3.5, 3.8, 5., 0.0001, 4.5,
        0.0001,
        0.0001, 1.5, 1.3, 0.8, 0.7, 0.9, 1., 1., 1., 1.,
    ])
    # choose random products from random product_groups 3 and from random locations
    unique_pg3 = df["PG_ID_3"].unique()
    random_pg_3 = random.choices(unique_pg3, k=int(len(unique_pg3)//1.4))
    random_products = []
    for pg3 in random_pg_3:
        unique_prods = df[df["PG_ID_3"] == pg3][
            "P_ID"
        ].unique()
        random_products += list(random.choices(unique_prods, k=int(len(unique_prods)//1.5)))
    prods_event_type_1 = random_products[: len(random_products) // 2]
    prods_event_type_2 = random_products[len(random_products) // 2:]
    unique_loc = df["L_ID"].unique()
    locs_event_type_2 = random.choices(unique_loc, k=int(len(unique_loc)//1.2))

    # construct data frame from calendar
    class PandasEventCalendar(AbstractHolidayCalendar):
        rules = events_list

    cal = PandasEventCalendar()
    df_events = (
        pd.DataFrame(
            cal.holidays(
                str(df["DATE"].min()),
                str(df["DATE"].max()),
                return_name=True,
            )
        )
        .rename(columns={0: "EVENT"})
        .reset_index()
        .rename(columns={"index": "DATE"})
    )

    df = df.merge(df_events, on="DATE", how="left")

    # add date_from and date_upto for every event
    df_events["date_from"] = pd.to_datetime(df_events["DATE"] - timedelta(days=10))
    df_events["date_upto"] = pd.to_datetime(df_events["DATE"] + timedelta(days=10))

    df_events_factors = df_events
    df_events_factors['DATE'] = [pd.date_range(s, e, freq='d') for s, e in
                                 zip(df_events["date_from"], df_events["date_upto"])]
    df_events_factors = df_events_factors.explode('DATE').drop(["date_from", "date_upto"], axis=1)
    df_events_factors["event_factor"] = 0.0

    df_events_factors.loc[df_events_factors["EVENT"] == "Christmas", "event_factor"] = christmas_type
    df_events_factors.loc[df_events_factors["EVENT"] == "Easter", "event_factor"] = easter_type

    df_events_factors.loc[df_events_factors["EVENT"] == "German_Unity", "event_factor"] = event_type_1
    df_events_factors.loc[df_events_factors["EVENT"] == "Labour_Day", "event_factor"] = event_type_1

    df_events_factors.loc[df_events_factors["EVENT"] == "Local_Holiday_0", "event_factor"] = event_type_2
    df_events_factors.loc[df_events_factors["EVENT"] == "Local_Holiday_1", "event_factor"] = event_type_2

    df_events_factors["EVENT_TYPE"] = np.nan
    df_events_factors.loc[df_events_factors["EVENT"].isin(
        ["Labour_Day", "German_Unity", "Christmas", "Easter"]
    ), "EVENT_TYPE"] = 1
    df_events_factors.loc[df_events_factors["EVENT"].isin(
        ["Local_Holiday_0", "Local_Holiday_1"]
    ), "EVENT_TYPE"] = 2
    del df_events_factors["EVENT"]
    df_events_factors = df_events_factors.groupby('DATE').max().reset_index()
    df = df.merge(df_events_factors, on=["DATE"], how="left")

    noise = np.random.normal(scale=0.1, size=len(df.loc[df["event_factor"].notna()]))
    df.loc[df["event_factor"].notna(), "event_factor"] += noise
    df["event_factor"] = df["event_factor"].fillna(0)

    df["LOG_LAMBDA"] = (
        df["LOG_LAMBDA"]
        .add(df["event_factor"])
        .where(
            (df["P_ID"].isin(prods_event_type_1)) &
            (df["EVENT_TYPE"] == 1),
            df["LOG_LAMBDA"],
        )
    )

    df["LOG_LAMBDA"] = (
        df["LOG_LAMBDA"]
        .add(df["event_factor"])
        .where(
            (df["P_ID"].isin(prods_event_type_2)) &
            (df["L_ID"].isin(locs_event_type_2)) &
            (df["EVENT_TYPE"] == 2),
            df["LOG_LAMBDA"],
        )
    )

    del df["event_factor"]
    del df["EVENT_TYPE"]

    return df
