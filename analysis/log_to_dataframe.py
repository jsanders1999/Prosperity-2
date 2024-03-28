
import json
import pandas as pd
from io import StringIO
import logging
import matplotlib.pyplot as plt
import numpy as np

dataFile = "data/tutorial_data.log"



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

def plot_price(product_history_df, product):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title(f"Price for {product}")

    # Plot the profit and loss for each product
    volume = np.array(product_history_df['quantity'])

    ax.scatter(product_history_df['timestamp'], product_history_df['price'], marker='d', s = volume,  c = "k")

    # Add a legend
    ax.legend()

    # Display the plot
    #plt.show()

def plot_market_orders(product_df, product):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title(f"Market Orders for {product}")

    # Plot the market orders for each product
    for i in range(1,3):
        #TODO: Fix NaN values
        
        bid_prices = np.array(product_df[f'bid_price_{i}'])
        timestamps = np.array(product_df['timestamp'])
        timestamps = timestamps[~np.isnan(bid_prices)]
        bid_prices = bid_prices[~np.isnan(bid_prices)]

        volumes = np.array(product_df[f'bid_volume_{i}'] )
        volumes = volumes[~np.isnan(volumes)]
        colors = np.zeros((len(volumes), 3))
        colors[:, 1] = 0.2+volumes/np.max(volumes)*0.6
        ax.scatter(timestamps, bid_prices, marker='^', s= volumes/2, c= colors, alpha=0.5)

        ask_prices = np.array(product_df[f'ask_price_{i}'])
        timestamps = np.array(product_df['timestamp'])
        timestamps = timestamps[~np.isnan(ask_prices)]
        ask_prices = ask_prices[~np.isnan(ask_prices)] 

        volumes = np.array(product_df[f'ask_volume_{i}'] )
        volumes = volumes[~np.isnan(volumes)]
        colors = np.zeros((len(volumes), 3))
        colors[:, 2] = 0.2+volumes/np.max(volumes)*0.6
        ax.scatter(timestamps, ask_prices, marker='v', s= volumes/2, c= colors, alpha=0.5)

    # Display the plot
    #plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    activities_df, trade_history_df = unpack_log_data(dataFile)
    product_activities_dfs = split_activities_df(activities_df)
    product_history_dfs = split_trade_history_df(trade_history_df)
    

    for key, product_df in product_activities_dfs.items():
        plot_market_orders(product_df, key)

    for key, product_df in product_history_dfs.items():
        plot_price(product_df, key)

    #plot_pnl_per_product(product_activities_dfs)
    plt.show()
    


