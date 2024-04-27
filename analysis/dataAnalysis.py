
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

def calc_decaying_moving_average(df, decay):
    moving_avg = np.zeros(len(df))
    moving_avg[0] = df[0]
    for i in range(1, len(df)):
        moving_avg[i] = (1-decay)*df[i] + decay*moving_avg[i-1]
    return moving_avg

def analyse_LOVETF_data():
    activities_paths = [f"data/ProvidedData/prices_round_3_day_{i}.csv" for i in [0,1,2]]
    trade_history_paths = [f"data/ProvidedData/trades_round_3_day_{i}_nn.csv" for i in [0,1,2]]

    fig_prem, ax_prem = plt.subplots()

    means = []
    stds = []

    fig_price, ax_price = plt.subplots(4,1, figsize=(15,10))
    
    for i, (activities_path, trade_history_path) in enumerate(zip(activities_paths, trade_history_paths)):
        activities_df, trade_history_df = read_csv_files(activities_path, trade_history_path)

        product_activities_dfs = split_activities_df(activities_df)
        product_history_dfs = split_trade_history_df(trade_history_df)

        weigh_mid_price_basket = calc_weighted_mid_price(product_activities_dfs["GIFT_BASKET"])
        weigh_mid_price_strawberries = calc_weighted_mid_price(product_activities_dfs["STRAWBERRIES"])
        weigh_mid_price_roses = calc_weighted_mid_price(product_activities_dfs["ROSES"])
        weigh_mid_price_chocolate = calc_weighted_mid_price(product_activities_dfs["CHOCOLATE"])

        moving_avg_basket = calc_decaying_moving_average(weigh_mid_price_basket, 0.97)
        moving_avg_strawberries = calc_decaying_moving_average(weigh_mid_price_strawberries, 0.95)
        moving_avg_roses = calc_decaying_moving_average(weigh_mid_price_roses, 0.95)
        moving_avg_chocolate = calc_decaying_moving_average(weigh_mid_price_chocolate, 0.97)

        

        weigh_mid_price_CSR = 4*weigh_mid_price_chocolate+6*weigh_mid_price_strawberries+weigh_mid_price_roses

        premium_signal = weigh_mid_price_basket-weigh_mid_price_CSR

        timestamps = product_activities_dfs["GIFT_BASKET"]['timestamp'] + i*1000000

        ax_price[0].plot(timestamps, weigh_mid_price_basket, label=f"Day {i}")
        ax_price[1].plot(timestamps, weigh_mid_price_strawberries, label=f"Day {i}")
        ax_price[2].plot(timestamps, weigh_mid_price_roses, label=f"Day {i}")
        ax_price[3].plot(timestamps, weigh_mid_price_chocolate, label=f"Day {i}")

        ax_price[0].plot(timestamps, moving_avg_basket, c='r', label=f"Moving Average Day {i}")
        ax_price[1].plot(timestamps, moving_avg_strawberries, c='r', label=f"Moving Average Day {i}")
        ax_price[2].plot(timestamps, moving_avg_roses, c='r', label=f"Moving Average Day {i}")
        ax_price[3].plot(timestamps, moving_avg_chocolate, c='r', label=f"Moving Average Day {i}")

        #set the titles of the subplots
        ax_price[0].set_title("GIFT_BASKET")
        ax_price[1].set_title("STRAWBERRIES")
        ax_price[2].set_title("ROSES")
        ax_price[3].set_title("CHOCOLATE")

        



        ax_prem.plot(timestamps, premium_signal, label=f"Day {i}")
        ax_prem.plot(timestamps, np.ones_like(premium_signal)*np.mean(premium_signal), c ='k', linewidth=0.5, label = f"Mean Day {i} = {np.mean(premium_signal)}")

        

        #store the mean of the premium signal
        means.append(np.mean(premium_signal))
        stds.append(np.std(premium_signal))

        plot_etf(product_activities_dfs)
    
    #calculate the combined mean and std of the premium signal
    combined_mean = np.average(means, weights = [1/std**2 for std in stds])
    combined_std = np.sqrt(1/np.sum([1/std**2 for std in stds]))

    ax_prem.plot(np.array([0, 3000000]), np.ones(2)*combined_mean, c='r', linewidth=0.5, label=f"Combined Mean = {combined_mean}")
    ax_prem.plot(np.array([0, 3000000]), np.ones(2)*(combined_mean+combined_std), c='r', linestyle='--', linewidth=0.5, label=f"Combined Mean+Std = {combined_mean+combined_std}")
    ax_prem.plot(np.array([0, 3000000]), np.ones(2)*(combined_mean-combined_std), c='r', linestyle='--', linewidth=0.5, label=f"Combined Mean-Std = {combined_mean-combined_std}")
    
    ax_prem.legend()
    print(f"Combined Mean: {combined_mean}")
    print(f"Combined Std: {combined_std}")

    plt.show()


    fig1, ax1 = plt.subplots()


    for key, activity_df in product_activities_dfs.items():
        histogram_mid_price_process(activity_df, key)
        #historgram_weighted_mid_price_process(activity_df, key)
        #fig1, ax1 = plot_market_orders(activity_df, key)
        if key == 'AMETHYSTS' or key=="STARFRUIT":
            bound = 1.45
        elif key == 'ORCHIDS':
            bound = 5
        else: 
            bound = 100
        plot_weighted_mid_price(activity_df, key, bound = bound, fig = fig1, ax = ax1)
        #trades_df = product_history_dfs[key]
        #plot_trades(trades_df, key, fig2, ax2)

    plt.show()

def plot_etf(product_activities_df, fig=None, ax=None):
    
    fig1, ax1 = plt.subplots()

    weigh_mid_price_basket = calc_weighted_mid_price(product_activities_df["GIFT_BASKET"])
    weigh_mid_price_strawberries = calc_weighted_mid_price(product_activities_df["STRAWBERRIES"])
    weigh_mid_price_roses = calc_weighted_mid_price(product_activities_df["ROSES"])
    weigh_mid_price_chocolate = calc_weighted_mid_price(product_activities_df["CHOCOLATE"])

    weigh_mid_price_CSR = 4*weigh_mid_price_chocolate+6*weigh_mid_price_strawberries+weigh_mid_price_roses

    timestamps = product_activities_df["GIFT_BASKET"]['timestamp']


    ax1.plot(timestamps, weigh_mid_price_basket, label='GIFT_BASKET')
    ax1.plot(timestamps, weigh_mid_price_CSR, label='4*C+6*S+1*R')
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(timestamps[1:], np.diff(weigh_mid_price_basket), label='GIFT_BASKET')
    ax2.plot(timestamps[1:], np.diff(weigh_mid_price_CSR), label='4*C+6*S+1*R')
    #ax2.plot(timestamps, weigh_mid_price_chocolate.diff(), label='CHOCOLATE')
    #ax2.plot(timestamps, weigh_mid_price_strawberries.diff(), label='STRAWBERRIES')
    #ax2.plot(timestamps, weigh_mid_price_roses.diff(), label='ROSES')

    premium_signal = weigh_mid_price_basket-weigh_mid_price_CSR

    analyse_premium_signal(premium_signal)

    fig4, ax4 = plt.subplots()
    ax4.plot(timestamps, weigh_mid_price_basket-weigh_mid_price_CSR, label='GIFT_BASKET-CSR')
    ax4.legend()

    print(np.std(np.diff(weigh_mid_price_basket-weigh_mid_price_CSR)))

    diff_basket = np.diff(weigh_mid_price_basket)
    diff_CSR = np.diff(weigh_mid_price_CSR)
    diff_chocolate = np.diff(weigh_mid_price_chocolate)
    diff_strawberries = np.diff(weigh_mid_price_strawberries)
    diff_roses = np.diff(weigh_mid_price_roses)


    norm_diff_basket = diff_basket#/np.std(diff_basket)
    norm_diff_CSR = diff_CSR#/np.std(diff_CSR)
    norm_diff_chocolate = diff_chocolate#/np.std(diff_chocolate)
    norm_diff_strawberries = diff_strawberries#/np.std(diff_strawberries)
    norm_diff_roses = diff_roses#/np.std(diff_roses)


    print(diff_basket)

    def cov(t):
        if t>0:
            m = np.array([norm_diff_basket[t:], norm_diff_chocolate[:-t], norm_diff_strawberries[:-t], norm_diff_roses[:-t], norm_diff_CSR[:-t]])
        elif t<0:
            m = np.array([norm_diff_basket[:t], norm_diff_chocolate[-t:], norm_diff_strawberries[-t:], norm_diff_roses[-t:], norm_diff_CSR[-t:]])
        else:
            m = np.array([norm_diff_basket, norm_diff_chocolate, norm_diff_strawberries, norm_diff_roses, norm_diff_CSR])
        cov_matrix = np.cov(m)
        return cov_matrix
    
    for t in range(-5, 5):
        print("")
        print(t)
        print(cov(t))
        

    ax2.legend()


def analyse_premium_signal(premium_signal):


    # #plot an FFT of the premium signal
    # fig1, ax1 = plt.subplots()
    # ax1.plot(premium_signal)
    # #fig2, ax2 = plt.subplots(2,1)
    # #ax2[0].plot(np.real(np.fft.fft(premium_signal)))
    # #ax2[1].plot(np.imag(np.fft.fft(premium_signal)))
    # #plt.show()

    stds = []

    diff_premium_signal = np.diff(premium_signal)
    fig, ax = plt.subplots()
    bins = np.linspace(np.floor(np.min(diff_premium_signal))-0.5, np.ceil(np.max(diff_premium_signal))+0.5, 5*(int(np.ceil(np.max(diff_premium_signal)-np.floor(np.min(diff_premium_signal)))+1)+1))
    ax.hist(np.diff(premium_signal), bins=bins, density=True, alpha=0.6, color='g')
    #fit a normal distribution to the diff of the premium signal
    from scipy.stats import norm
    mu, std = norm.fit(np.diff(premium_signal))
    x = np.linspace(np.min(np.diff(premium_signal)), np.max(np.diff(premium_signal)), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label='fit: mu=%.2f, std=%.2f' % (mu, std))
    #fit a t-distribution to the diff of the premium signal
    from scipy.stats import t
    df, loc, scale = t.fit(np.diff(premium_signal))
    p = t.pdf(x, df, loc, scale)
    ax.plot(x, p, 'r', linewidth=2, label='fit: df=%.2f, loc=%.2f, scale=%.2f' % (df, loc, scale))
    ax.legend()

    equilibrium = np.mean(premium_signal)
    print(f"Equilibrium: {equilibrium}")

    p_list = np.array(list(range(1,1000)))
    diff_list = []
    mu_list = []
    for p in p_list:
        stds.append(np.std(premium_signal[p:]- premium_signal[:-p]))
        diff_list += [[premium_signal[p:]- premium_signal[:-p]]]
        mu_list += [np.mean(premium_signal[:-p]- equilibrium)]

    #fig, ax = plt.subplots()
    slope_list = []
    for p in p_list:
        # fig, ax = plt.subplots()
        # ax.plot(premium_signal[p:]-equilibrium, premium_signal[:-p]-equilibrium, '.', markersize=0.5)
        #do linear regression on this data
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(premium_signal[p:]-equilibrium, premium_signal[:-p]-equilibrium)
        # print(f"R^2: {r_value**2}")
        # print(f"Slope: {slope}")
        # print(f"Intercept: {intercept}")
        # print(f"p-value: {p_value}")
        # print(f"std_err: {std_err}")
        slope_list.append(slope)
        # x = np.linspace(np.min(premium_signal[p:]-equilibrium), np.max(premium_signal[p:]-equilibrium), 100)
        # ax.plot(x, slope*x+intercept, c='r', linewidth=0.5)
        # plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(p_list, slope_list)
    #fit f(p) = nu^p to the slope_list data
    from scipy.optimize import curve_fit
    def func(p, nu):
        return nu**p
    res = curve_fit(func, p_list, slope_list, sigma =p_list/10000, p0=[0.9])
    print(res)
    ax.plot(p_list, func(p_list, *res[0]) , c='r')   
    ax.plot(p_list, func(p_list, 0.99879548), c='b', linewidth=0.5)


    def varfunc(p, sigma, nu):
        return sigma*np.sqrt((1-nu**(2*p))/(1-nu**2))

    from scipy.optimize import curve_fit
    res = curve_fit(varfunc, p_list, stds, sigma = (10000-p_list)/10000, p0=[6, 0.9])
    print(res)

    fig2, ax2 = plt.subplots()
    ax2.scatter(p_list, stds)
    ax2.plot(p_list, varfunc(p_list, *res[0]), c='r')
    ax2.plot(p_list, res[0][0]*np.sqrt(p_list), c='b', linewidth=0.5)

    return

def analyse_coconut_coupon_data():

    activities_path = "data/ProvidedData/prices_round_4_day_1.csv"
    trade_history_path = "data/ProvidedData/trades_round_4_day_1_nn.csv"
    activities_df, trade_history_df = read_csv_files(activities_path, trade_history_path)

    product_activities_dfs = split_activities_df(activities_df)
    product_history_dfs = split_trade_history_df(trade_history_df)

    wmid_prices_coconuts = calc_weighted_mid_price(product_activities_dfs['COCONUT'])
    wmid_prices_coupons = calc_weighted_mid_price(product_activities_dfs['COCONUT_COUPON'])

    timestamps = product_activities_dfs['COCONUT']['timestamp']

    delta_wmid_prices_coconuts = np.diff(wmid_prices_coconuts)
    delta_wmid_prices_coupons = np.diff(wmid_prices_coupons)

    fig0, ax0 = plt.subplots()
    ax0.scatter(delta_wmid_prices_coconuts, delta_wmid_prices_coupons)

    #fit a multivariate normal distribution to the data
    from scipy.stats import multivariate_normal
    data = np.array([delta_wmid_prices_coconuts, delta_wmid_prices_coupons]).T
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    print(mean)
    print(cov)
    rv = multivariate_normal(mean=mean, cov=cov)


    fig, ax = plt.subplots()
    ax.plot(wmid_prices_coconuts, wmid_prices_coupons)

    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(timestamps, wmid_prices_coconuts, label='COCONUT')
    ax1[1].plot(timestamps, wmid_prices_coupons, label='COUPONS')
    #plot the decay moving average of the weighted mid prices
    decay_coconut = 0.95
    decay_coupon = 0.97
    moving_avg_coconuts = calc_decaying_moving_average(wmid_prices_coconuts, decay_coconut)
    moving_avg_coupons = calc_decaying_moving_average(wmid_prices_coupons, decay_coupon)
    ax1[0].plot(timestamps, moving_avg_coconuts, c='r', label=f"Moving Average Decay={decay_coconut}")
    ax1[1].plot(timestamps, moving_avg_coupons, c='r', label=f"Moving Average Decay={decay_coupon}")
    ax1[0].legend()
    ax1[1].legend()


    fig2, ax2 = plt.subplots()

    # covariance = []
    # t_lim = 50
    # for t in range(-t_lim, t_lim):
    #     print(t)
    #     if t>0:
    #         covariance.append(np.cov([delta_wmid_prices_coconuts[:-t], delta_wmid_prices_coupons[t:]]))
    #     elif t<0:
    #         covariance.append(np.cov([delta_wmid_prices_coconuts[-t:], delta_wmid_prices_coupons[:t]]))
    #     else:
    #         covariance.append(np.cov([delta_wmid_prices_coconuts, delta_wmid_prices_coupons]))
    # covariance = np.array(covariance)

    # ax2.plot(range(-t_lim, t_lim), covariance[:,1,0])

    # for key, activity_df in product_activities_dfs.items():
    #     #histogram_mid_price_process(activity_df, key)
    #     historgram_weighted_mid_price_process(activity_df, key)

    #     fig2, ax2 = plot_weighted_mid_price(activity_df, key, bound = 100)
    #     trades_df = product_history_dfs[key]
    #     plot_trades(trades_df, key, fig2, ax2)


    # for key, product_df in product_history_dfs.items():
    #     #plot_trades(product_df, key, fig2, ax2)
    #     plot_trades_difference(product_df, product_activities_dfs[key], key, bound=100)

    plt.show()

    







if __name__ == "__main__":

    # analyse_coconut_coupon_data()
    # raise Error
    #plot_orchid_data()

    #logging.basicConfig(level=logging.INFO)


    #log_path = "data/Round1Logs/round1_final.log"

    #log_path = "data/Round2Logs/null_trader_data.log"   
    #log_path = "data/Round2Logs/trader_AS_2.log"
    #log_path = "data/Round2Logs/trader_ASO_-1.log"
    #log_path = "data/Round2Logs/arbitrage.log"
    #log_path = "data/Round2Logs/round2_final.log"
    #log_path = "data/Round3Logs/deltas0_5.5test.log"
    #log_path = "data/Round3Logs/langevin_0.0_6_12_5.5_6_7.log"
    #log_path = "data/Round3Logs/round3_final.log"
    log_path = "data/Round4Logs/round4final.log"

    activities_df, trade_history_df = read_log_file(log_path)
    #raise Error

    #analyse_LOVETF_data()

    
    # activities_path = "data/ProvidedData/prices_round_3_day_0.csv"
    # trade_history_path = "data/ProvidedData/trades_round_3_day_0_nn.csv"
    # activities_df, trade_history_df = read_csv_files(activities_path, trade_history_path)


    product_activities_dfs = split_activities_df(activities_df)
    product_history_dfs = split_trade_history_df(trade_history_df)

    #trades_csv_output_path = "analysis/backtestData/trades.csv"
    #trade_history_df.to_csv(trades_csv_output_path, index=False, sep=';')

    #histogram_asks_bids(product_activities_dfs['AMETHYSTS'], 'AMETHYSTS')
    #histogram_trades(product_history_dfs['AMETHYSTS'], product_activities_dfs['AMETHYSTS'], 'AMETHYSTS')
    #histogram_trades(product_history_dfs['STARFRUIT'], product_activities_dfs['STARFRUIT'], 'STARFRUIT')
    #histogram_trades(product_history_dfs['ORCHIDS'], product_activities_dfs['ORCHIDS'], 'ORCHIDS')
    # histogram_trades(product_history_dfs['GIFT_BASKET'], product_activities_dfs['GIFT_BASKET'], 'GIFT_BASKET')
    # histogram_trades(product_history_dfs['CHOCOLATE'], product_activities_dfs['CHOCOLATE'], 'CHOCOLATE')
    # histogram_trades(product_history_dfs['STRAWBERRIES'], product_activities_dfs['STRAWBERRIES'], 'STRAWBERRIES')
    # histogram_trades(product_history_dfs['ROSES'], product_activities_dfs['ROSES'], 'ROSES')


    for key, activity_df in product_activities_dfs.items():
        #histogram_mid_price_process(activity_df, key)
        #historgram_weighted_mid_price_process(activity_df, key)
        #fig1, ax1 = plot_market_orders(activity_df, key)
        if key == 'AMETHYSTS' or key=="STARFRUIT":
            bound = 1.45
        elif key == 'ORCHIDS':
            bound = 5
        else: 
            bound = 100
        fig2, ax2 = plot_weighted_mid_price(activity_df, key, bound = bound)
        trades_df = product_history_dfs[key]
        plot_trades(trades_df, key, fig2, ax2)


    for key, product_df in product_history_dfs.items():
        #plot_trades(product_df, key, fig2, ax2)
        if key == 'AMETHYSTS' or key=="STARFRUIT":
            bound = 1.45
        elif key == 'ORCHIDS':
            bound = 5
        else: 
            bound = 100
        plot_trades_difference(product_df, product_activities_dfs[key], key, bound)

    plot_pnl_per_product(product_activities_dfs)
    plt.show()

    


