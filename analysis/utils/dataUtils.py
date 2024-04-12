import json
import pandas as pd
from io import StringIO
import logging
import matplotlib.pyplot as plt
import numpy as np
import jsonpickle as jp


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