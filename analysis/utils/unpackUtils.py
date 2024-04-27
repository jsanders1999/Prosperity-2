import json
import pandas as pd
from io import StringIO
import logging
import matplotlib.pyplot as plt
import numpy as np
import jsonpickle as jp

def read_log_file(log_path):
    activities_df, trade_history_df = unpack_log_data(log_path)
    return activities_df, trade_history_df

def read_csv_files(activities_path, trade_history_path):
    activities_df = pd.read_csv(activities_path, sep=';')
    trade_history_df = pd.read_csv(trade_history_path, sep=';')
    return activities_df, trade_history_df

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
    sandbox_logs_str = sandbox_logs_str.replace('"sandboxLog": "",\n', '')
    sandbox_logs_str = sandbox_logs_str.replace('\\\\"', '"')
    sandbox_logs_str = sandbox_logs_str.replace('\\"', '"')
    sandbox_logs_str = sandbox_logs_str.replace('\\n', '\n')
    sandbox_logs_str = sandbox_logs_str.replace('"{', '{')
    sandbox_logs_str = sandbox_logs_str.replace('}"', '}')
    activities_log_str = activities_log_str.replace('Activities log:\n', '')
    trade_history_str = trade_history_str.replace('Trade History:\n', '')

    log_list = sandbox_logs_str.split('\n}\n{\n')
    log_list[0] = log_list[0].replace('{\n', '')
    log_list[-1] = log_list[-1].replace('\n}', '')

    log_dict = {}
    for element in log_list:
        # Parse the input string into a dictionary
        parsed_data = json.loads("{"+element+"}")

        # Extracting lambdaLog and timestamp
        lambda_log = parsed_data['lambdaLog']
        timestamp = parsed_data['timestamp']

        # Creating a dictionary with timestamp as key and lambdaLog as value
        result = {timestamp: lambda_log}
        log_dict.update(result)
        
    print(log_dict)


    
    #sandbox_logs = pd.DataFrame(json.loads('[' + sandbox_logs_str.replace('\n', '') + ']'))
    trade_history = trade_history_string_to_dataframe(trade_history_str)
    #logging.info(trade_history)

    activities_log = activities_string_to_dataframe(activities_log_str)
    #logging.info(activities_log)

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