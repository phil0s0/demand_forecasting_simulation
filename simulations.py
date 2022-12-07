import sys
import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from IPython import embed

from base_demand import \
    simulate_products, \
    simulate_product_groups, \
    simulate_locations, \
    simulate_dates, \
    simulate_basic_demand
from optional_demand import \
    simulate_coupling_demand
from seasonalities import \
    simulate_trend,\
    simulate_weekday_profile, \
    simulate_yearly_seasonality, \
    simulate_monthly_profile, \
    simulate_school_holidays
from prices import \
    simulate_normal_price, \
    simulate_promotions, \
    simulate_reduced_prices, \
    simulate_price_model
from events import simulate_events
from variance import simulate_inv_r


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    n_products = 200
    n_locations = 20
    n_pg3 = 20

    date_from = '2019-10-01'
    date_upto = '2022-09-30'

    df = simulate_products(
        n_products=n_products,
        p_id_shift=1
    )
    df = simulate_product_groups(
        df,
        n_pg3=n_pg3,
        pg_id_shift=1
    )
    # generate product groups with different number of products
    drop_indices = np.random.choice(df.index, n_products//10, replace=False)
    df = df.drop(drop_indices)

    df = simulate_normal_price(df)

    df = simulate_locations(
        df,
        n_locations=n_locations,
        l_id_shift=1
    )

    df = simulate_dates(
        df,
        date_from=date_from,
        date_upto=date_upto
    )

    df = simulate_basic_demand(df)

    df = simulate_coupling_demand(df, n_couples=20, max_products=3)

    df = simulate_trend(df)

    df = simulate_weekday_profile(df)

    df = simulate_school_holidays(df)

    # payday effect
    df = simulate_monthly_profile(df)

    df = simulate_yearly_seasonality(df)

    df["LOG_LAMBDA"].clip(upper=5.0, inplace=True)

    df = simulate_events(df)

    df = simulate_promotions(df)

    df = simulate_reduced_prices(df)

    df = simulate_price_model(df)

    # simulate non-assignment situations (e.g., out-of-stock)
    # for unknown assignments set to zero sales instead of dropping
    df = df.sample(frac=0.95, random_state=1)

    df["LAMBDA"] = np.exp(df["LOG_LAMBDA"])

    # simulate variance and draw sales from negative binomial distribution
    df = simulate_inv_r(df)

    del df["LOG_LAMBDA"]
    del df["ELASTICITY"]

    # df.to_csv("../train_data.csv", index=False)
    # df.to_parquet("../train_data.parquet.gzip", compression='gzip')
    del df["LAMBDA"]

    df["SALES"].hist(log=True)
    plt.savefig("sales.pdf")
    plt.clf()

    df_train = df.loc[df['DATE']<='2022-03-31']
    df_train.reset_index(inplace=True, drop=True)
    df_test = df.loc[df['DATE']>'2022-03-31']
    df_test.reset_index(inplace=True, drop=True)
    # df_train.to_csv("train.csv", index=False)
    df_train.to_parquet("train.parquet.gzip", compression='gzip')
    # df_test[["P_ID", "L_ID", "DATE", "SALES"]].to_csv("test_results.csv", index=False)
    df_test[["P_ID", "L_ID", "DATE", "SALES"]].to_parquet("test_results.parquet.gzip", compression='gzip')
    del df_test["SALES"]
    # df_test.to_csv("test.csv", index=False)
    df_test.to_parquet("test.parquet.gzip", compression='gzip')

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
