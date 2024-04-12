
import json
import pandas as pd
from io import StringIO
import logging
import matplotlib.pyplot as plt
import numpy as np
import jsonpickle as jp

from utils.dataUtils import calc_weighted_mid_price, calc_weighted_mid_price_ask_bid, calc_moving_average, calc_filtered_weighted_mid_price, calc_predicted_price
from utils.unpackUtils import read_log_file, read_csv_files, split_activities_df, split_trade_history_df
from utils.plottingUtils import plot_market_orders, plot_weighted_mid_price, plot_trades, plot_trades_difference, plot_pnl_per_product, histogram_asks_bids, histogram_trades, histogram_mid_price_process, historgram_weighted_mid_price_process


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    #dataFile = "data/constant_orders_2_1,55_tutorial_data.log"
    #dataFile = "data/null_tutorial_data.log"
    #dataFile = "data/temp.log"
    #dataFile = "data/speculative_1,55_2,55_4_tut_data.log"

    #log_path = "data/TutorialLogs/constant_orders_2&4_tutorial_data.log"

    #log_path = "data/Round1Logs/speculative_STRF_1,54_2,54_4_AMTH_2_4_4.log"
    #log_path = "data/Round1Logs/temp.log"
    #log_path = "data/Round1Logs/speculative_STRF_2,5_3,5_4_2_AMTH_2_4_4.log"
    #log_path = "data/Round1Logs/speculative_STRF_2,3_3,5_2,1_3,5_1_1_AMTH_2_4_10.log"
    #log_path = "data/Round1Logs/speculative_STRF_2,3_3,5_2,1_3,5_6_0_AMTH_2_4_11.log"
    #log_path = "data/Round1Logs/speculative_STRF_2,3_3,5_2,1_3,5_6_1_AMTH_2_4_3.log"
    log_path = "data/Round1Logs/round1_final.log"
    activities_df, trade_history_df = read_log_file(log_path)

    #activities_path = "data/ProvidedData/prices_round_1_day_-2.csv"
    #trade_history_path = "data/ProvidedData/trades_round_1_day_-2_nn.csv"
    #activities_df, trade_history_df = read_csv_files(activities_path, trade_history_path)

    product_activities_dfs = split_activities_df(activities_df)
    product_history_dfs = split_trade_history_df(trade_history_df)

    #trades_csv_output_path = "analysis/backtestData/trades.csv"
    #trade_history_df.to_csv(trades_csv_output_path, index=False, sep=';')

    #histogram_asks_bids(product_activities_dfs['AMETHYSTS'], 'AMETHYSTS')
    histogram_trades(product_history_dfs['AMETHYSTS'], product_activities_dfs['AMETHYSTS'], 'AMETHYSTS')
    histogram_trades(product_history_dfs['STARFRUIT'], product_activities_dfs['STARFRUIT'], 'STARFRUIT')

    for key, activity_df in product_activities_dfs.items():
        #histogram_mid_price_process(activity_df, key)
        #historgram_weighted_mid_price_process(activity_df, key)
        fig, ax = plot_market_orders(activity_df, key)
        plot_weighted_mid_price(activity_df, key)
        #trades_df = product_history_dfs[key]
        #plot_trades(trades_df, key, fig, ax)


    for key, product_df in product_history_dfs.items():
       plot_trades(product_df, key)
       plot_trades_difference(product_df, product_activities_dfs[key], key)

    plot_pnl_per_product(product_activities_dfs)
    plt.show()

    


