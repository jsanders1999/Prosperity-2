
import json
import pandas as pd
from io import StringIO
import logging
import matplotlib.pyplot as plt
import numpy as np

def activities_string_to_dataframe(csv_string):
    # Use StringIO to convert the string to a file-like object
    csv_file = StringIO(csv_string)
    
    # Use pd.read_csv() to read the file-like object into a DataFrame
    df = pd.read_csv(csv_file, sep=';')
    
    return df

def trade_history_string_to_dataframe(json_string):
    # Parse the JSON string into a Python list of dictionaries
    data_list = json.loads(json_string)
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data_list)
    
    return df


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

    # Display the plot
    #plt.show()

def split_trade_history_df_by_buyer_seller(trade_history_df):
    seller_grouped_trade_history_df = trade_history_df.groupby("seller")
    

    # Create a dictionary to store separate DataFrames for each product
    grouped_dfs = {}

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
            
    return grouped_dfs

def plot_trades(product_history_df, product, fig = None, ax = None):
    # Create a figure and axis
    if ax is None or fig is None:
        fig, ax = plt.subplots()
        ax.set_title(f"Price for {product}")

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

def historgram_weighted_mid_price_process(product_df, product):
    # Create a figure and axis
    fig, ax = plt.subplots()


    weighted_mid_prices = calc_weighted_mid_price(product_df)
    dS = np.diff(weighted_mid_prices)
    ax.set_title(f"Weighted Mid Price change histogram for {product}\n mean = {dS.mean()}, std = {dS.std()}")
    bins = np.linspace(dS.min()-0.25, dS.max()+0.25,  int(20*(dS.max() - dS.min())+2))
    ax.hist(dS[1:], bins=bins ,density=True, histtype='step', color='black')
    # plot normal distibution over the histogram
    x = np.linspace(dS.min(), dS.max(), 100)
    pdf = np.exp(-x**2 / (2 * dS.std()**2)) / (dS.std() * np.sqrt(2 * np.pi))
    ax.plot(x, pdf, color='red')
    # Display the plot
    #plt.show()

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




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    dataFile = "data/weighted_mid_tutorial_data.log"

    activities_df, trade_history_df = unpack_log_data(dataFile)
    product_activities_dfs = split_activities_df(activities_df)
    product_history_dfs = split_trade_history_df(trade_history_df)

    histogram_asks_bids(product_activities_dfs['AMETHYSTS'], 'AMETHYSTS')

    for key, activity_df in product_activities_dfs.items():
        #histogram_mid_price_process(product_df, key)
        #historgram_weighted_mid_price_process(product_df, key)
        fig, ax = plot_market_orders(activity_df, key)
        #trades_df = product_history_dfs[key]
        #plot_trades(trades_df, key, fig, ax)


    for key, product_df in product_history_dfs.items():
        plot_trades(product_df, key)

    plot_pnl_per_product(product_activities_dfs)
    plt.show()

    


