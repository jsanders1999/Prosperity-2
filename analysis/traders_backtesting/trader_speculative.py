from analysis.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import jsonpickle as jp

#TODO
# See if order sorting is needed or if it is already sorted
# 1. Print the logs in a way that can easily be unpacked into a dict/dataframe
# 2. Start storing revelvant variables in the traderData variable
# 3. Implement a function to calculate the acceptable price for each product
# 4. Implement a backteter to test the strategy on historical data

MAX_MIDPOINT_MEM = 50

class Trader:
    
    def run(self, state: TradingState):
        loggingDataDict = {}
        if len(state.traderData)==0:
            traderDataDict = {}
        else:
            traderDataDict = jp.decode(state.traderData)

        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        loggingDataDict["traderData"] =  state.traderData
        loggingDataDict["listings"] = state.listings
        #loggingDataDict["own_trades"] = state.own_trades
        #loggingDataDict["order_depths"] = state.order_depths
        #loggingDataDict["market_trades"] = state.market_trades
        loggingDataDict["position"] = state.position
        loggingDataDict["observations"] = state.observations

        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            position = fetch_current_position(state.position, product)
            orders: List[Order] = []

            loggingDataDict[product] = {}

            if product not in traderDataDict.keys():
                traderDataDict[product] = {}

            # Set the acceptable price and position limit for each product
            if product == "STARFRUIT":
                
                #Calculate the weighted mid price
                weighted_mid_price = calc_weighted_mid_price(order_depth.buy_orders, order_depth.sell_orders)
                #Calculate the filtered mid price and the speculative price
                if "weighted_mid_price" not in traderDataDict[product].keys():
                    # Run at the zeroth iteration
                    filtered_mid_price = weighted_mid_price
                    speculative_price = weighted_mid_price
                else:
                    prev_filtered_mid_price = traderDataDict[product]["filtered_mid_price"][-1]
                    filtered_mid_price = filter_price_extrema(weighted_mid_price, prev_filtered_mid_price)
                    speculative_price = speculative_price_starfruit(weighted_mid_price, prev_filtered_mid_price)
                
                #Put the weighted mid price in the logging data
                loggingDataDict[product]["weighted_mid_price"] = weighted_mid_price
                loggingDataDict[product]["filtered_mid_price"] = filtered_mid_price
                loggingDataDict[product]["speculative_price"] = speculative_price
                #Store the weighted mid price and the filtered mid price in the traderDataDict
                traderDataDict = store_weighted_mid_price(traderDataDict, product, weighted_mid_price)
                traderDataDict = store_filtered_mid_price(traderDataDict, product, filtered_mid_price)
                
                acceptable_price_buy = speculative_price #can be changed to filtered_mid_price or weighted_mid_price
                acceptable_price_sell = filtered_mid_price
                    
                #TODO: put constants in an external dictionary
                position_limit = 20
                bound1 = 1.54
                bound2 = bound1+1
                cutoff = 4
                orders = trade_constant_two_levels_two_prices(product, orders, position, acceptable_price_buy, acceptable_price_sell, position_limit, bound1, bound2, cutoff)
                #orders =  trade_constant_two_levels(product, orders, position, acceptable_price, position_limit, bound1, bound2, cutoff)

                #bound = 1.55
                #orders = trade_constant(product, orders, position, acceptable_price, position_limit, bound)

                #Calculate the moving average of the weighted mid price
                #window = 3
                #moving_average = np.mean(traderDataDict[product]["weighted_mid_price"][-window:])

            elif product == "AMETHYSTS":

                acceptable_price = 10000; 
                position_limit = 20
                
                bound1 = 2
                bound2 = 4
                cutoff = 4
                orders =  trade_constant_two_levels(product, orders, position, acceptable_price, position_limit, bound1, bound2, cutoff)

                #bound = 2
                #orders = trade_constant(product, orders, position, acceptable_price, position_limit, bound)
        
            result[product] = orders
        
        print(jp.encode(loggingDataDict))
    
        traderData = jp.encode(traderDataDict) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    
    
def store_weighted_mid_price(traderData, product, weighted_mid_price):
    if "weighted_mid_price" in traderData[product].keys():
        traderData[product]["weighted_mid_price"] += [weighted_mid_price]
        #keep the length under a certain limit
        if len(traderData[product]["weighted_mid_price"]) > MAX_MIDPOINT_MEM:
            traderData[product]["weighted_mid_price"] = traderData[product]["weighted_mid_price"][1:]
    else:
        traderData[product]["weighted_mid_price"] = [weighted_mid_price]
    return traderData

def store_filtered_mid_price(traderData, product, filtered_mid_price):
    if "filtered_mid_price" in traderData[product].keys():
        traderData[product]["filtered_mid_price"] += [filtered_mid_price]
        #keep the length under a certain limit
        if len(traderData[product]["filtered_mid_price"]) > MAX_MIDPOINT_MEM:
            traderData[product]["filtered_mid_price"] = traderData[product]["filtered_mid_price"][1:]
    else:
        traderData[product]["filtered_mid_price"] = [filtered_mid_price]
    return traderData

def filter_price_extrema(price_now, price_prev, bound = 1.45):
    print(f"price_now: {price_now}, price_prev: {price_prev}")
    delta = price_now - price_prev
    if delta <= -bound:
        return price_prev - 1
    elif delta >= bound:
        return price_prev + 1
    else:
        return price_now

def speculative_price_starfruit(price_now, price_prev, bound = 1.45):
    #ONLY WORKS FOR STARFRUIT
    #TODO: round the probabilities so it does not have a downwards trend
    probability_array = np.array([[0.01101101, 0.11661662, 0.04804805],
                                    [0.11611612, 0.45545546, 0.0975976],
                                    [0.04854855, 0.0975976 , 0.00900901]])
    delta = round(price_now - price_prev)
    if delta <=-1:
        return price_now + probability_array[0,:]@[-1, 0, 1]/np.sum(probability_array[0,:])
    elif delta == 0:
        return price_now + probability_array[1,:]@[-1, 0, 1]/np.sum(probability_array[1,:])
    elif delta >= 1:
        return price_now + probability_array[2,:]@[-1, 0, 1]/np.sum(probability_array[2,:])
    else:
        return price_now


def trade_best_bids_asks(product, orders, order_depth, position, acceptable_price, position_limit, offset):
    """ loop through the ordered sell orders and buy orders and place orders if the price is acceptable """

    print("Acceptable price : " + str(acceptable_price))
    print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

    n_sell_orders = len(order_depth.sell_orders)
    n_buy_orders = len(order_depth.buy_orders)
    # loop through the ordered sell orders and buy orders and place orders if the price is acceptable
    for i in range(max(n_sell_orders, n_buy_orders)):
        if len(order_depth.sell_orders) != 0 and i < n_sell_orders:
            best_ask, best_ask_amount = get_ith_sell_order(order_depth.sell_orders, i)
            #print("Best ask: ", best_ask, best_ask_amount)
            if int(best_ask) < acceptable_price - offset:
                # position + amount <= position_limit
                print(f"buy i = {i}")
                print("Best ask price: ", best_ask)
                print("Best ask amount: ", best_ask_amount)
                print("Amount: ", min(position_limit - position, -best_ask_amount))
                amount = min(position_limit - position, -best_ask_amount)
                position += amount
                orders = place_buy_order(orders, product, best_ask, amount)

        if len(order_depth.buy_orders) != 0 and i < n_buy_orders:
            best_bid, best_bid_amount = get_ith_buy_order(order_depth.buy_orders, i)
            #print("Best bid: ", best_bid, best_bid_amount)
            if best_bid > acceptable_price + offset:
                #position - amount >= -position_limit
                print(f"sell i = {i}")
                print("Best bid price: ", best_bid)
                print("Best bid amount: ", best_bid_amount)
                print("Amount: ", min(position_limit + position, best_bid_amount))
                amount = min(position_limit + position, best_bid_amount)
                orders = place_sell_order(orders, product, best_bid, amount)
                position -= amount

    return orders

def trade_constant(product, orders, position, accteptable_price, position_limit, bound):
    buy_amount = position_limit - position
    sell_amount = position_limit + position
    #buy_price = accteptable_price - 2
    #sell_price = accteptable_price + 2
    #two_buy_amount = min(6, buy_amount)#buy_amount - four_buy_amount
    #two_sell_amount = min(6, sell_amount)#sell_amount - four_sell_amount
    #four_buy_amount = buy_amount-two_buy_amount#round(buy_amount / 3)
    #four_sell_amount = sell_amount-two_sell_amount#round(sell_amount / 3)
    orders = place_buy_order(orders, product, round(accteptable_price - bound), buy_amount)
    orders = place_sell_order(orders, product, round(accteptable_price + bound), sell_amount)
    return orders

def trade_constant_two_levels(product, orders, position, accteptable_price, position_limit, bound1, bound2, ratio):
    buy_amount = position_limit - position
    sell_amount = position_limit + position

    buy_price_1 = round(accteptable_price - bound1)
    sell_price_1 = round(accteptable_price + bound1)
    buy_price_2 = round(accteptable_price - bound2)
    sell_price_2 = round(accteptable_price + bound2)
    if buy_amount<4:
        buy_amount_2 = buy_amount
    else:
        buy_amount_2 = 0#min(int(np.ceil(ratio*buy_amount)), buy_amount)

    if sell_amount<4:
        sell_amount_2 = sell_amount
    else:
        sell_amount_2 = 0
    #sell_amount_2 = min(int(np.ceil(ratio*sell_amount)), sell_amount)
    buy_amount_1 = buy_amount - buy_amount_2
    sell_amount_1 = sell_amount - sell_amount_2
    orders = place_buy_order(orders, product, buy_price_2, buy_amount_2)
    orders = place_buy_order(orders, product, buy_price_1, buy_amount_1)
    orders = place_sell_order(orders, product, sell_price_2, sell_amount_2)
    orders = place_sell_order(orders, product, sell_price_1, sell_amount_1)
    return orders
    
def trade_constant_two_levels_two_prices(product, orders, position, acceptable_price_buy, acceptable_price_sell, position_limit, bound1, bound2, ratio):
    buy_amount = position_limit - position
    sell_amount = position_limit + position

    buy_price_1 = round(acceptable_price_buy - bound1)
    sell_price_1 = round(acceptable_price_sell + bound1)
    buy_price_2 = round(acceptable_price_buy - bound2)
    sell_price_2 = round(acceptable_price_sell + bound2)
    if buy_amount<4:
        buy_amount_2 = buy_amount
    else:
        buy_amount_2 = 0#min(int(np.ceil(ratio*buy_amount)), buy_amount)

    if sell_amount<4:
        sell_amount_2 = sell_amount
    else:
        sell_amount_2 = 0
    #sell_amount_2 = min(int(np.ceil(ratio*sell_amount)), sell_amount)
    buy_amount_1 = buy_amount - buy_amount_2
    sell_amount_1 = sell_amount - sell_amount_2
    orders = place_buy_order(orders, product, buy_price_2, buy_amount_2)
    orders = place_buy_order(orders, product, buy_price_1, buy_amount_1)
    orders = place_sell_order(orders, product, sell_price_2, sell_amount_2)
    orders = place_sell_order(orders, product, sell_price_1, sell_amount_1)
    return orders


def fetch_current_position(position, product):
    if product in position.keys():
        return position[product]
    else:
        return 0
    

def place_buy_order(orders, product, ask, amount):
    """
    A utility function to place a buy order. 
    Written to make the code more readable
    amount should be positive
    """
    #print("BUY ", product, str(-amount) + "x", ask)
    orders.append(Order(product, ask, amount))
    return orders

def place_sell_order(orders, product, bid, amount):
    """
    A utility function to place a sell order. 
    Written to make the code more readable
    amount should be positive
    """
    #print("SELL ", product,  str(amount) + "x", bid)
    orders.append(Order(product, bid, -amount))
    return orders

def calc_mid_price(buy_orders, sell_orders):
    #return the mid price of the best buy and sell orders
    best_buy_order = get_best_buy_order(buy_orders)
    best_sell_order = get_best_sell_order(sell_orders)
    return (best_buy_order[0] + best_sell_order[0]) / 2

def calc_weighted_mid_price(buy_orders, sell_orders):
    #return the weighted mid price of the best buy and sell orders
    sum_price_mul_volume = 0
    sum_volume = 0
    for price, volume in buy_orders.items():
        sum_price_mul_volume += price * volume
        sum_volume += volume
    for price, volume in sell_orders.items():
        #Note: volume is negative for sell orders
        sum_price_mul_volume += price *(- volume)
        sum_volume += (-volume)
    return (sum_price_mul_volume / sum_volume)

def get_best_sell_order(sell_orders):
    #return the sell order with the lowest price
    sorted_sell_orders = sorted(list(sell_orders.items()), key=lambda x: x[0], reverse=False)
    return sorted_sell_orders[0]

def get_best_buy_order(buy_orders):
    #return the buy order with the highest price
    sorted_buy_orders =  sorted(list(buy_orders.items()), key=lambda x: x[0], reverse=True)
    return sorted_buy_orders[0]

def get_ith_sell_order(sell_orders, i):
    #return the sell order with the lowest price

    sorted_sell_orders = sorted(list(sell_orders.items()), key=lambda x: x[0], reverse=False)
    return sorted_sell_orders[i]

def get_ith_buy_order(buy_orders, i):
    #return the buy order with the highest price
    sorted_buy_orders =  sorted(list(buy_orders.items()), key=lambda x: x[0], reverse=True)
    return sorted_buy_orders[i]