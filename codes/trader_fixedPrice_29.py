from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np

#TODO
# 1. Print the logs in a way that can easily be unpacked into a dict/dataframe
# 2. Start storing revelvant variables in the traderData variable
# 3. Implement a function to calculate the acceptable price for each product
# 4. Implement a backteter to test the strategy on historical data

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        #print("Listings: " + str(state.listings))
        #print("Own trades: " + str(state.own_trades))
        #print("Order depths: " + str(state.order_depths))
        #print("Market trades: " + str(state.market_trades))
        print("Position: " + str(state.position))
        #print("Observations: " + str(state.observations))

        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Set the acceptable price and position limit for each product
            if product == "STARFRUIT":
                acceptable_price = 4970;  # Participant should calculate this value
                position_limit = 20
            elif product == "AMETHYSTS":
                acceptable_price = 10000;  # Participant should calculate this value
                position_limit = 20
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            #Get the current position (buy - sell from previous trades)
            position = fetch_current_position(state.position, product)
                
            n_sell_orders = len(order_depth.sell_orders)
            n_buy_orders = len(order_depth.buy_orders)
            # loop through the ordered sell orders and buy orders and place orders if the price is acceptable
            for i in range(max(n_sell_orders, n_buy_orders)):
                if len(order_depth.sell_orders) != 0 and i < n_sell_orders:
                    best_ask, best_ask_amount = get_ith_sell_order(order_depth.sell_orders, i)
                    #print("Best ask: ", best_ask, best_ask_amount)
                    if int(best_ask) < acceptable_price:
                        # position + amount <= position_limit
                        print(f"buy i = {i}")
                        print("Best ask price: ", best_ask)
                        print("Best ask amount: ", best_ask_amount)
                        print("Amount: ", min(position_limit - position, -best_ask_amount))
                        amount = min(position_limit - state.position[product], -best_ask_amount)
                        position += amount
                        orders = place_buy_order(orders, product, best_ask, amount)

                if len(order_depth.buy_orders) != 0 and i < n_buy_orders:
                    best_bid, best_bid_amount = get_ith_buy_order(order_depth.buy_orders, i)
                    #print("Best bid: ", best_bid, best_bid_amount)
                    if int(best_bid) > acceptable_price:
                        #position - amount >= -position_limit
                        print(f"sell i = {i}")
                        print("Best bid price: ", best_bid)
                        print("Best bid amount: ", best_bid_amount)
                        print("Amount: ", min(position_limit + position, best_bid_amount))
                        amount = min(position_limit + position, best_bid_amount)
                        orders = place_sell_order(orders, product, best_bid, amount)
                        position -= amount
            
            result[product] = orders
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    

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