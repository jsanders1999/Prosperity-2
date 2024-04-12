
import json
import pandas as pd
from io import StringIO
import logging
import matplotlib.pyplot as plt
import numpy as np
import jsonpickle as jp

#TODO: Check if histogram probabilities are correct

def activities_string_to_dataframe(csv_string):
    csv_file = StringIO(csv_string)
    df = pd.read_csv(csv_file, sep=';')
    return df

def trade_history_string_to_dataframe(json_string):
    data_list = json.loads(json_string)
    df = pd.DataFrame(data_list)
    return df

def log_data_to_dataframe(log_data_file_path):
    #TODO: Implement
    return

def unpack_log_data(log_data_file_path):
    # Split the log data into three sections
    log_data = ""
    for line in open(log_data_file_path):
        log_data += line
    sandbox_logs_str, activities_log_str, trade_history_str = log_data.split('\n\n\n\n')

    # Remove the section headers
    sandbox_logs_str = sandbox_logs_str.replace('Sandbox logs:\n', '')
    activities_log_str = activities_log_str.replace('Activities log:\n', '')
    trade_history_str = trade_history_str.replace('Trade History:\n', '')
    
    #sandbox_logs = pd.DataFrame(json.loads('[' + sandbox_logs_str.replace('\n', '') + ']'))
    trade_history = trade_history_string_to_dataframe(trade_history_str)
    logging.info(trade_history)

    activities_log = activities_string_to_dataframe(activities_log_str)
    logging.info(activities_log)

    return activities_log, trade_history

def split_activities_df(activities_df):
    grouped_activities_df = activities_df.groupby("product")

    # Create a dictionary to store separate DataFrames for each product
    product_dfs = {}

    # Iterate over each group and store it in the dictionary
    for product, group in grouped_activities_df:
        product_dfs[product] = group.reset_index(drop=True)

        # Access the separate DataFrames for each product
        for product, product_df in product_dfs.items():
            logging.info(f"DataFrame for {product}:")
            logging.info(product_df)
            logging.info("\n")
            
    return product_dfs

def split_trade_history_df(trade_history_df):
    grouped_trade_history_df = trade_history_df.groupby("symbol")

    # Create a dictionary to store separate DataFrames for each product
    product_dfs = {}

    # Iterate over each group and store it in the dictionary
    for product, group in grouped_trade_history_df:
        product_dfs[product] = group.reset_index(drop=True)

        # Access the separate DataFrames for each product
        for product, product_df in product_dfs.items():
            logging.info(f"DataFrame for {product}:")
            logging.info(product_df)
            logging.info("\n")
            
    return product_dfs

def plot_pnl_per_product(product_dfs):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the profit and loss for each product
    for product, product_df in product_dfs.items():
        ax.plot(product_df['timestamp'], product_df['profit_and_loss'], marker='.', markersize = 0.5, linestyle="None", label=product)

    # Add a legend
    ax.legend()

    return fig, ax

def split_trade_history_df_by_buyer_seller(trade_history_df):
    seller_grouped_trade_history_df = trade_history_df.groupby("seller")
    

    # Create a dictionary to store separate DataFrames for each product
    grouped_dfs = {}

    if "SUBMISSION" in seller_grouped_trade_history_df.groups.keys():
        # Iterate over each group and store it in the dictionary
        for seller, group in seller_grouped_trade_history_df:
            if seller=="SUBMISSION":
                grouped_dfs["SOLD"] = group.reset_index(drop=True)
            else:
                temp_df = group.reset_index(drop=True)

        buyer_grouped_trade_history_df = temp_df.groupby("buyer")
        for buyer, group in buyer_grouped_trade_history_df:
            if buyer=="SUBMISSION":
                grouped_dfs["BOUGHT"] = group.reset_index(drop=True)
            else:
                grouped_dfs["OTHERS"] = group.reset_index(drop=True)
    else:
        grouped_dfs["OTHERS"] = trade_history_df.reset_index(drop=True)
            
    return grouped_dfs

def plot_position(trade_history_df, product):
    fig, ax = plt.subplots()
    ax.set_title(f"Position for {product}")

    timestamps = trade_history_df['timestamp']

    grouped_trades_df = split_trade_history_df_by_buyer_seller(trade_history_df)
    if "SOLD" in grouped_trades_df.keys() and "BOUGHT" in grouped_trades_df.keys():
        bought_df = grouped_trades_df["BOUGHT"]
        sold_df = grouped_trades_df["SOLD"]

        bought_timestamps = bought_df['timestamp']
        sold_timestamps = sold_df['timestamp']

        position = np.zeros(len(timestamps))

        for i, time in enumerate(timestamps):
            position[i] = np.sum(bought_df['quantity'][bought_timestamps<=time]) - np.sum(sold_df['quantity'][sold_timestamps<=time])
        ax.plot(timestamps, position, marker='.', markersize = 1.5, linestyle="None", label="position")

    return



def plot_trades(product_history_df, product, fig = None, ax = None):
    # Create a figure and axis
    if ax is None or fig is None:
        fig, ax = plt.subplots()
        ax.set_title(f"Trades for {product}")

    # Plot the profit and loss for each product
    grouped_trade_dfs = split_trade_history_df_by_buyer_seller(product_history_df)

    for key, trade_df in grouped_trade_dfs.items():
        trades = trade_df['price']
        volume = trade_df['quantity']
        timestamp = trade_df['timestamp']
        if key == "SOLD":
            color = "red"
            marker = "$O$"
            alpha = 0.9
        elif key == "BOUGHT":
            color = "blue"
            marker = "$X$"
            alpha = 0.9
        else:
            color = "teal"
            marker = '$+$'
            alpha = 0.5
        ax.scatter(timestamp,trades, marker=marker, s = 5*volume,  c = color, label = key, alpha = alpha)

    # Add a legend
    ax.legend()

    plot_position(product_history_df, product)

    # Display the plot
    #plt.show()
    return fig, ax

def plot_trades_difference(product_history_df, product_activity_df, product, fig = None, ax = None):
    # Create a figure and axis
    if ax is None or fig is None:
        fig, ax = plt.subplots()
        ax.set_title(f"Trades for {product}")

    # Plot the profit and loss for each product
    grouped_trade_dfs = split_trade_history_df_by_buyer_seller(product_history_df)

    weighted_mid_price = calc_weighted_mid_price(product_activity_df)
    filtered_weighted_mid_price, _ = calc_filtered_weighted_mid_price(weighted_mid_price, 2, 1.45)


    for key, trade_df in grouped_trade_dfs.items():

        trades = trade_df['price']

        if product == "AMETHYSTS":
            values = 10000*np.ones(trades.shape)
        else:
            values = filtered_weighted_mid_price #np.round(filtered_weighted_mid_price-0.5)+0.5
            relevant_inds  = []
            trade_timestamps = trade_df['timestamp']
            activity_timestamps = product_activity_df['timestamp']
            for i, timestamp in enumerate(trade_timestamps):
                relevant_inds += [list(activity_timestamps).index(timestamp)]
            values = values[relevant_inds]

        volume = trade_df['quantity']
        timestamp = trade_df['timestamp']
        if key == "SOLD":
            color = "red"
            marker = "$O$"
            alpha = 0.9
        elif key == "BOUGHT":
            color = "blue"
            marker = "$X$"
            alpha = 0.9
        else:
            color = "teal"
            marker = '$+$'
            alpha = 0.5
        ax.scatter(timestamp,trades-values, marker=marker, s = 5*volume,  c = color, label = key, alpha = alpha)

    # Add a legend
    ax.legend()

    # Display the plot
    #plt.show()
    return fig, ax

def calc_weighted_mid_price(product_df):
    prices_mul_volumes = np.zeros((len(product_df), 6))
    volumes = np.zeros((len(product_df), 6))

    for i in range(0,3):
        prices_mul_volumes[:,i] = product_df[f'bid_price_{i+1}']*product_df[f'bid_volume_{i+1}']
        prices_mul_volumes[:,i+3] =  product_df[f'ask_price_{i+1}']*product_df[f'ask_volume_{i+1}']
        volumes[:,i] = product_df[f'bid_volume_{i+1}']
        volumes[:,i+3] = product_df[f'ask_volume_{i+1}']
    timestamps = product_df['timestamp']
    weighted_mid_prices = np.nansum(prices_mul_volumes, axis = 1)/np.nansum(volumes, axis = 1)
    return weighted_mid_prices

def calc_weighted_mid_price_ask_bid(product_df):
    prices_mul_volumes = np.zeros((len(product_df), 6))
    volumes = np.zeros((len(product_df), 6))

    for i in range(0,3):
        prices_mul_volumes[:,i] = product_df[f'bid_price_{i+1}']*product_df[f'bid_volume_{i+1}']
        prices_mul_volumes[:,i+3] =  product_df[f'ask_price_{i+1}']*product_df[f'ask_volume_{i+1}']
        volumes[:,i] = product_df[f'bid_volume_{i+1}']
        volumes[:,i+3] = product_df[f'ask_volume_{i+1}']
    timestamps = product_df['timestamp']
    weighted_mid_prices_bid = np.nansum(prices_mul_volumes[:,0:3], axis = 1)/np.nansum(volumes[:,0:3], axis = 1)
    weighted_mid_prices_ask = np.nansum(prices_mul_volumes[:,3:6], axis = 1)/np.nansum(volumes[:,3:6], axis = 1)
    weighted_mid_prices = (weighted_mid_prices_bid + weighted_mid_prices_ask)/2
    return weighted_mid_prices

def calc_moving_average(weighted_mid_prices, window):
    moving_avg = weighted_mid_prices.copy()
    for i in range(window):
        moving_avg[i] = np.mean(weighted_mid_prices[:i+1])
    moving_avg[window-1:] = np.convolve(weighted_mid_prices, np.ones(window)/window, mode='valid')
    return moving_avg

def calc_filtered_weighted_mid_price(weighted_mid_prices, window, bound, offset = 1.0):
    #moving_avg = calc_moving_average(weighted_mid_prices, window)
    filtered_weighted_mid_prices = weighted_mid_prices.copy()
    different_inds = []
    for i in range(1, len(weighted_mid_prices)):
        if np.abs(weighted_mid_prices[i]- filtered_weighted_mid_prices[i-1]) > bound:
            filtered_weighted_mid_prices[i] = filtered_weighted_mid_prices[i-1] + np.sign(weighted_mid_prices[i]- filtered_weighted_mid_prices[i-1])*offset
            different_inds.append(i)
    return filtered_weighted_mid_prices, different_inds

def calc_predicted_price(prices):
    #Bases on toturial data:
    probability_array = np.array([[0.01101101, 0.11661662, 0.04804805],
                                    [0.11611612, 0.45545546, 0.0975976],
                                    [0.04854855, 0.0975976 , 0.00900901]])

    #Based on provided data round 1:
    #probability_array = np.array([[0.00993532, 0.1005201,  0.04554244],
    #                            [0.1010202,  0.46589318, 0.1090218 ],
    #                            [0.04504234, 0.1095219,  0.0135027 ]])
    
    print(np.sum(probability_array))
    predicted_prices = prices.copy()
    for i in range(1, len(prices)):
        curr_price = prices[i]
        prev_price = prices[i-1]
        curr_delta = np.round(curr_price - prev_price)
        if curr_delta <= -1 :
            predicted_prices[i] = curr_price + probability_array[0,:]@[-1, 0, 1]/np.sum(probability_array[0,:])
        elif curr_delta == 0:
            predicted_prices[i] = curr_price + probability_array[1,:]@[-1, 0, 1]/np.sum(probability_array[1,:])
        elif curr_delta >= 1:
            predicted_prices[i] = curr_price + probability_array[2,:]@[-1, 0, 1]/np.sum(probability_array[2,:])

    print("predicted prices", np.linalg.norm(predicted_prices[:-1] - prices[1:]))
    print("normal prices", np.linalg.norm(prices[:-1] - prices[1:]))
    return predicted_prices
    



def plot_market_orders(product_df, product):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title(f"Market Orders for {product}")

    # plot the mid prices as a line:
    mid_prices = product_df['mid_price']
    timestamps = product_df['timestamp']
    ax.plot(timestamps, mid_prices, marker = ".", markersize = 0.1, linewidth = 0.1, color='black')

    weighted_mid_prices = calc_weighted_mid_price(product_df)
    ax.plot(timestamps, weighted_mid_prices, marker = "s", markersize = 0.5, linewidth = 0.1, color='orange')
    # Plot the market orders for each product
    for i in range(1,4):
        #TODO: Fix NaN values
        
        bid_prices = np.array(product_df[f'bid_price_{i}'])
        timestamps = np.array(product_df['timestamp'])
        timestamps = timestamps[~np.isnan(bid_prices)]
        bid_prices = bid_prices[~np.isnan(bid_prices)]

        volumes = np.array(product_df[f'bid_volume_{i}'] )
        volumes = volumes[~np.isnan(volumes)]
        colors = np.zeros((len(volumes), 3))
        colors[:, 1] = 0.2+volumes/np.max(volumes)*0.6
        if i==1:
            ax.scatter(timestamps, bid_prices, marker='v', s= volumes/2, c= colors, alpha=0.5, label = "bid")
        else:
            ax.scatter(timestamps, bid_prices, marker='v', s= volumes/2, c= colors, alpha=0.5)


        ask_prices = np.array(product_df[f'ask_price_{i}'])
        timestamps = np.array(product_df['timestamp'])
        timestamps = timestamps[~np.isnan(ask_prices)]
        ask_prices = ask_prices[~np.isnan(ask_prices)] 

        volumes = np.array(product_df[f'ask_volume_{i}'] )
        volumes = volumes[~np.isnan(volumes)]
        colors = np.zeros((len(volumes), 3))
        colors[:, 2] = 0.2+volumes/np.max(volumes)*0.6
        if i==1:
            ax.scatter(timestamps, ask_prices, marker='^', s= volumes/2, c= colors, alpha=0.5, label = "ask")
        else:
            ax.scatter(timestamps, ask_prices, marker='^', s= volumes/2, c= colors, alpha=0.5)

        ax.legend()


    # Display the plot
    #plt.show()
    return fig, ax

def plot_weighted_mid_price(product_df, product, window = 3, bound = 1.45):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title(f"Mid prices for {product}")

    # plot the mid prices as a line:
    mid_prices = product_df['mid_price']
    timestamps = product_df['timestamp']
    #ax.plot(timestamps, mid_prices, marker = ".", markersize = 0.1, linewidth = 0.1, color='black')

    weighted_mid_prices = calc_weighted_mid_price(product_df)
    ax.plot(timestamps, weighted_mid_prices, marker = "s", markersize = 0.5, linewidth = 0.1, color='orange')

    #weighted_mid_prices_ask_bid = calc_weighted_mid_price_ask_bid(product_df)   
    #ax.plot(timestamps, weighted_mid_prices_ask_bid, marker = "s", markersize = 0.5, linewidth = 0.1, color='blue')

    moving_avg_w_m = calc_moving_average(weighted_mid_prices, window)
    #moving_avg_w_m_ask_bid = calc_moving_average(weighted_mid_prices_ask_bid, window)

    #ax.plot(timestamps[window-1:], moving_avg_w_m, marker = "s", markersize = 0.5, linewidth = 0.2, color='red')
    #ax.plot(timestamps[window-1:], moving_avg_w_m_ask_bid, marker = "s", markersize = 0.5, linewidth = 0.2, color='green')

    filtered_weighted_mid_prices, different_inds = calc_filtered_weighted_mid_price(weighted_mid_prices, window, bound)
    
    ax.plot(timestamps, filtered_weighted_mid_prices, marker = "x", markersize = 0.8, linewidth = 0.05, color='red')
    ax.plot(timestamps[different_inds], filtered_weighted_mid_prices[different_inds], marker = "$O$", markersize = 15, linestyle = "None", color='yellow')

    predicted_prices = calc_predicted_price(filtered_weighted_mid_prices)

    ax.plot(timestamps, predicted_prices, marker = "+", markersize = 3, linewidth = 0.05, color='purple')
        
    # if product == "AMETHYSTS":
    #     print(f"{weighted_mid_prices}")
    #     print(f"{np.mean(weighted_mid_prices-10000)=}")
    #     print(f"{np.std(weighted_mid_prices-10000)=}")

    #     #print(f"{weighted_mid_prices_ask_bid}")
    #     #print(f"{np.mean(weighted_mid_prices_ask_bid-10000)=}")
    #     #print(f"{np.std(weighted_mid_prices_ask_bid-10000)=}")
    #     fig, ax = plt.subplots()
    #     ax.set_title(f"Weighted Mid Price histogram for {product}")
    #     bins = np.linspace(-5.5, 5.5, 60)#12)
    #     ax.hist(weighted_mid_prices-10000, bins=bins, density=True, histtype='step', color='black', label = "weighted mid price")
    #     #ax.hist(weighted_mid_prices_ask_bid-10000, bins=bins, density=True, histtype='step', color='blue', label = "weighted mid price ask bid")
    #     ax.hist(filtered_weighted_mid_prices-10000, bins=bins, density=True, histtype='step', color='red', label = "filtered weighted mid price")
    #     ax.legend()

        
def histogram_mid_price_process(product_df, product):
    # Create a figure and axis
    fig, ax = plt.subplots()
    

    # plot the mid prices as a line:
    mid_prices = product_df['mid_price']
    dS = mid_prices.diff()

    ax.set_title(f"Mid Price change histogram for {product}\n mean = {dS.mean()}, std = {dS.std()}")

    #ax.plot(timestamps[1:], dS[1:], marker = ".", markersize = 0.1, linewidth = 0.1, color='black')
    bins = np.linspace(dS.min()-0.25, dS.max()+0.25,  int(2*(dS.max() - dS.min())+2))
    ax.hist(dS[1:], bins=bins ,density=True, histtype='step', color='black')
    # plot normal distibution over the histogram
    x = np.linspace(dS.min(), dS.max(), 100)
    pdf = np.exp(-x**2 / (2 * dS.std()**2)) / (dS.std() * np.sqrt(2 * np.pi))
    ax.plot(x, pdf, color='red')
    # Display the plot
    #plt.show()

def historgram_weighted_mid_price_process(product_df, product, window = 2, bound = 1.45):
    # Create a figure and axis
    fig, ax = plt.subplots()


    weighted_mid_prices = calc_weighted_mid_price(product_df)
    dS = np.diff(weighted_mid_prices)
    ax.set_title(f"Weighted Mid Price change histogram for {product}\n mean = {dS.mean()}, std = {dS.std()}")
    bins = np.linspace(dS.min()-0.25, dS.max()+0.25,  int(20*(dS.max() - dS.min())+2))
    ax.hist(dS[1:], bins=bins ,density=True, histtype='step', color='black')
    bins = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    ax.hist(dS[1:], bins=bins ,density=True, histtype='step', color='black')
    # plot normal distibution over the histogram
    x = np.linspace(dS.min(), dS.max(), 100)
    pdf = np.exp(-x**2 / (2 * dS.std()**2)) / (dS.std() * np.sqrt(2 * np.pi))
    ax.plot(x, pdf, color='red')
    # Display the plot
    #plt.show()
    auto_corr = np.correlate(dS, dS, mode='full')
    fig, ax = plt.subplots()    
    ax.set_title(f"Autocorrelation of change in weighted mid price for {product}")
    ax.plot(auto_corr)

    filtered_weighted_mid_price, different_inds = calc_filtered_weighted_mid_price(weighted_mid_prices, window, bound)

    data = filtered_weighted_mid_price #weighted_mid_prices

    dS_1 = data[1:] - data[:-1]
    #dS_2 = weighted_mid_prices[2:] - weighted_mid_prices[:-2]
    fig, ax = plt.subplots()
    ax.set_title(f"2D histogram of change in weighted mid price for {product}")
    ax.set_xlabel("dS_t")
    ax.set_ylabel("dS_{t-1}")
    bins= np.linspace(-1.5, 1.5, 4)#[-1.5, -0.5, 0.5, 1.5]
    h, x_edges, y_edges, img = ax.hist2d( dS_1[1:], dS_1[:-1], bins=bins, density=True, cmap = 'Blues')
    print(f"{product} {h=}")


def histogram_asks_bids(product_df, product):
    fig, ax = plt.subplots()
    ax.set_title(f"Ask quote histogram for {product}")
    ask_prices = np.concatenate([np.array(product_df['ask_price_1']), np.array(product_df['ask_price_2']), np.array(product_df['ask_price_3'])])
    print(ask_prices.shape)
    ask_prices = ask_prices[~np.isnan(ask_prices)]
    ask_volumes = np.concatenate([np.array(product_df['ask_volume_1']), np.array(product_df['ask_volume_2']), np.array(product_df['ask_volume_3'])])
    ask_volumes = ask_volumes[~np.isnan(ask_volumes)]
    h, xedges, yedges, img = ax.hist2d(ask_prices, ask_volumes, bins=8, density=True, cmap = 'Blues')
    fig.colorbar(img, ax = ax)

    fig, ax = plt.subplots()
    ax.set_title(f"Bid quote histogram for {product}")
    bid_prices = np.concatenate([np.array(product_df['bid_price_1']), np.array(product_df['bid_price_2']), np.array(product_df['bid_price_3'])])
    bid_prices = bid_prices[~np.isnan(bid_prices)]
    bid_volumes = np.concatenate([np.array(product_df['bid_volume_1']), np.array(product_df['bid_volume_2']), np.array(product_df['bid_volume_3'])])
    bid_volumes = bid_volumes[~np.isnan(bid_volumes)]
    h, xedges, yedges, img = ax.hist2d(bid_prices, bid_volumes, bins=8, density=True, cmap = 'Blues')
    fig.colorbar(img, ax = ax)

def histogram_trades(trades_df, activity_df, product, window = 2, bound=1.45):
    fig, ax = plt.subplots()
    ax.set_title(f"Trade histogram for {product}")
    trades = trades_df['price']
    trade_timestamps = trades_df['timestamp']
    if product == "AMETHYSTS":
        value = 10000*np.ones(trades.shape)
        activity_timestamps = activity_df['timestamp']
    else:
        activity_timestamps = activity_df['timestamp']
        weighted_mid_price = calc_weighted_mid_price(activity_df)
        #mid_price = np.array(activity_df['mid_price'])
        #filtered_weighted_mid_price, _ = calc_filtered_weighted_mid_price(weighted_mid_price, window, bound)
        value = weighted_mid_price #np.round(weighted_mid_price-0.5)+0.5 #filtered_weighted_mid_price
        relevant_inds  = []
        for i, timestamp in enumerate(trade_timestamps):
            relevant_inds += [list(activity_timestamps).index(timestamp)]
        value = value[relevant_inds]
    print(trades.shape)
    print(value.shape)
    difference = trades - value
    volumes = trades_df['quantity']
    bins = (np.linspace(np.floor(difference.min())-0.5, np.ceil(difference.max())+0.5, 10*(round(np.ceil(difference.max()) - np.floor(difference.min()))+1)+1),
            np.linspace(volumes.min()-0.5, volumes.max()+0.5, round(volumes.max() - volumes.min())+2))
    h, xedges, yedges, img = ax.hist2d(difference, volumes, bins=bins, density=True, cmap = 'Blues')
    print(f"{h=}")
    fig.colorbar(img, ax = ax)

    fig, ax = plt.subplots()
    ax.set_title(f"Trade histogram for {product}")
    probs = np.histogram(difference, bins=xedges, density=False, weights=volumes)[0]/(np.array(activity_timestamps)[-1]/100)
    #probs = h@(yedges[1:]-0.5)/np.array(activity_timestamps)[-1]
    print(f"for {product=} {probs=}")
    print(f"{xedges=}")
    ax.plot((xedges[:-1]+xedges[1:])/2, probs)

    h, xedges, yedges = np.histogram2d(difference, volumes, bins=bins, density=False)

    ax.scatter((xedges[:-1]+xedges[1:])/2, np.sum(h*(yedges[1:]-0.5), axis = 1)/(np.array(activity_timestamps)[-1]/100), color = 'red')


def read_log_file(log_path):
    activities_df, trade_history_df = unpack_log_data(log_path)
    return activities_df, trade_history_df

def read_csv_files(activities_path, trade_history_path):
    activities_df = pd.read_csv(activities_path, sep=';')
    trade_history_df = pd.read_csv(trade_history_path, sep=';')
    return activities_df, trade_history_df


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
    log_path = "data/Round1Logs/speculative_STRF_2,3_3,5_2,1_3,5_6_1_AMTH_2_4_3.log"
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

    


