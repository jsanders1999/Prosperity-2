from datamodel import OrderDepth, UserId, TradingState, Order
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
                weighted_mid_price = calc_weighted_mid_price(order_depth.buy_orders, order_depth.sell_orders)
                loggingDataDict[product]["weighted_mid_price"] = weighted_mid_price
                #TODO: make seperate function
                if "weighted_mid_price" in traderDataDict[product].keys():
                    traderDataDict[product]["weighted_mid_price"] += [weighted_mid_price]
                    #keep the length under a certain limit
                    if len(traderDataDict[product]["weighted_mid_price"]) > MAX_MIDPOINT_MEM:
                        traderDataDict[product]["weighted_mid_price"] = traderDataDict[product]["weighted_mid_price"][1:]
                else:
                    traderDataDict[product]["weighted_mid_price"] = [weighted_mid_price]
                window = 4
                moving_average = np.mean(traderDataDict[product]["weighted_mid_price"][-window:])
                acceptable_price = moving_average
                position_limit = 20
                offset = 1 #Optimize this parameter
                bound = 1.55
                #orders = trade_best_bids_asks(product, orders, order_depth, position, acceptable_price, position_limit, offset)
                orders = trade_constant(product, orders, position, acceptable_price, position_limit, bound)
            elif product == "AMETHYSTS":
                acceptable_price = 10000; 
                position_limit = 20
                offset = 0 
                bound = 2
                #orders = trade_best_bids_asks(product, orders,  order_depth, position, acceptable_price, position_limit, offset)
                orders = trade_constant(product, orders, position, acceptable_price, position_limit, bound)
        
            result[product] = orders
        
        print(jp.encode(loggingDataDict))
    
        traderData = jp.encode(traderDataDict) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData


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