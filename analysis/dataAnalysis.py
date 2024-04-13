
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


def plot_orchid_data():
    orchid_path = "data/ProvidedData/prices_round_2_day_-1.csv"
    df = pd.read_csv(orchid_path, sep=';')

    fig1, ax1 = plt.subplots()
    ax1.plot(df['timestamp'], df['ORCHIDS'])

    fig2, ax2 = plt.subplots()
    ax2.plot(df['timestamp'], df['EXPORT_TARIFF'], label='EXPORT_TARIFF')
    ax2.plot(df['timestamp'], df['IMPORT_TARIFF'], label='IMPORT_TARIFF')
    ax2.plot(df['timestamp'], df['TRANSPORT_FEES'], label='TRANSPORT_FEES')
    ax2.legend()

    fig3, ax3 = plt.subplots(2,1)
    ax3[0].plot(df['timestamp'], df['SUNLIGHT'], label='SUNLIGHT')
    ax3[1].plot(df['timestamp'], df['HUMIDITY'], label='HUMIDITY')
    ax3[0].legend()
    ax3[1].legend()

    fig4, ax4 = plt.subplots(2,1)
    ax4[0].plot(df['timestamp'], df['SUNLIGHT'].diff(), label='d/dt SUNLIGHT')
    ax4[1].plot(df['timestamp'], df['HUMIDITY'].diff(), label='d/dt HUMIDITY')
    ax4[0].legend()
    ax4[1].legend()

    fig5, ax5 = plt.subplots(2,1)
    ax5[0].plot(df['timestamp'], df['SUNLIGHT'].diff().diff(), label='d^2/dt^2 SUNLIGHT')
    ax5[1].plot(df['timestamp'], df['HUMIDITY'].diff().diff(), label='d^2/dt^2 HUMIDITY')
    ax5[0].legend()
    ax5[1].legend()

    diff_orchids = np.array(df['ORCHIDS'].diff())
    diff_sunlight = np.array(df['SUNLIGHT'].diff())
    diff_humidity = np.array(df['HUMIDITY'].diff())

    sunlight_no_nan = df["SUNLIGHT"][~np.isnan(diff_sunlight)]
    humidity_no_nan = df["HUMIDITY"][~np.isnan(diff_humidity)]

    diff_orchids = diff_orchids[~np.isnan(diff_orchids)]
    diff_sunlight = diff_sunlight[~np.isnan(diff_sunlight)]
    diff_humidity = diff_humidity[~np.isnan(diff_humidity)]

    diff_diff_sunlight = np.diff(diff_sunlight)
    diff_diff_humidity = np.diff(diff_humidity)
    no_nan_diff_orchids = diff_orchids[1:]


    fig6, ax6 = plt.subplots(1,2)
    ax6[0].scatter(sunlight_no_nan, diff_orchids, label='ORCHIDS')
    ax6[1].scatter(humidity_no_nan, diff_orchids, label='ORCHIDS')
    ax6[0].legend()
    ax6[1].legend()

    fig7, ax7 = plt.subplots(1,2)
    ax7[0].hist2d(np.diff(diff_sunlight), no_nan_diff_orchids, bins=(50,50))
    ax7[1].hist2d(np.diff(diff_humidity), no_nan_diff_orchids, bins=(50,50))

    fig8, ax8 = plt.subplots()

    ax8.plot(df['timestamp'], df['ORCHIDS'], c= "k", label='ORCHIDS')
    ax8.plot(df['timestamp'], df["ORCHIDS"]+df['IMPORT_TARIFF']+df["TRANSPORT_FEES"], c="r", label='BUY_COSTS')
    ax8.plot(df['timestamp'], df["ORCHIDS"]-df['EXPORT_TARIFF']-df["TRANSPORT_FEES"], c="b", label='SELL_GAINS')
    ax8.legend()

    fig9, ax9 = plt.subplots()
    ax9.plot(np.convolve(diff_orchids, diff_orchids[::-1], mode='same'), label='ORCHIDS')
    ax9.plot(range(len(np.convolve(diff_sunlight, diff_sunlight[::-1], mode='same'))), np.zeros_like(np.convolve(diff_orchids, diff_orchids[::-1], mode='same')), label='SUNLIGHT')

    fig10, ax10 = plt.subplots()
    bins = np.linspace(np.floor(np.min(diff_orchids))-0.5, np.ceil(np.max(diff_orchids))+0.5, 20*(int(np.ceil(np.max(diff_orchids)-np.floor(np.min(diff_orchids)))+1)+1))
    print(bins)
    ax10.hist(diff_orchids, bins=bins, density=True, alpha=0.6, color='g')

    
    plt.show()

    print(df.head())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    #log_path = "data/Round1Logs/round1_final.log"

    #log_path = "data/Round2Logs/null_trader_data.log"   
    #log_path = "data/Round2Logs/trader_AS_2.log"
    log_path = "data/Round2Logs/trader_AS_2.log"
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
    histogram_trades(product_history_dfs['ORCHIDS'], product_activities_dfs['ORCHIDS'], 'ORCHIDS')

    for key, activity_df in product_activities_dfs.items():
        #histogram_mid_price_process(activity_df, key)
        #historgram_weighted_mid_price_process(activity_df, key)
        #fig1, ax1 = plot_market_orders(activity_df, key)
        if key == 'AMETHYSTS' or key=="STARFRUIT":
            bound = 1.45
        elif key == 'ORCHIDS':
            bound = 5
        fig2, ax2 = plot_weighted_mid_price(activity_df, key, bound = bound)
        trades_df = product_history_dfs[key]
        plot_trades(trades_df, key, fig2, ax2)


    for key, product_df in product_history_dfs.items():
       #plot_trades(product_df, key, fig2, ax2)
       plot_trades_difference(product_df, product_activities_dfs[key], key)

    plot_pnl_per_product(product_activities_dfs)
    plt.show()

    


