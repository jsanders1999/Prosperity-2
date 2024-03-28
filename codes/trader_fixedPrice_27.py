from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np

REPEATS = 1

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
            if product == "STARFRUIT":
                acceptable_price = 4970;  # Participant should calculate this value
                position_limit = 20
            elif product == "AMETHYSTS":
                acceptable_price = 10000;  # Participant should calculate this value
                position_limit = 20
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            for i in range(REPEATS):
                #todo: make function
                if product in state.position.keys():
                    position = state.position[product]
                else:
                    position = 0

                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = get_best_sell_order(order_depth.sell_orders)
                    #print("Best ask: ", best_ask, best_ask_amount)
                    if int(best_ask) < acceptable_price:
                        # position + amount <= position_limit
                        print("Best ask amount: ", best_ask_amount)
                        print("Amount: ", min(position_limit - position, -best_ask_amount))
                        amount = min(position_limit - state.position[product], -best_ask_amount)
                        orders = place_buy_order(orders, product, best_ask, amount)
                        
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = get_best_buy_order(order_depth.buy_orders)
                    #print("Best bid: ", best_bid, best_bid_amount)
                    if int(best_bid) > acceptable_price:
                        #position - amount >= -position_limit
                        print("Best bid amount: ", best_bid_amount)
                        print("Amount: ", min(position_limit + position, best_bid_amount))
                        amount = min(position_limit + position, best_bid_amount)
                        orders = place_sell_order(orders, product, best_bid, amount)
            
            result[product] = orders
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    

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