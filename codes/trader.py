from codes.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10;  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    place_buy_order(orders, product, best_ask, best_ask_amount)
                    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    place_sell_order(orders, product, best_bid, best_bid_amount)
                    
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    

def place_buy_order(orders, product, ask, amount):
    """
    A utility function to place a buy order. 
    Written to make the code more readable
    """
    print("BUY", str(-amount) + "x", ask)
    orders.append(Order(product, ask, -amount))

def place_sell_order(orders, product, bid, amount):
    """
    A utility function to place a sell order. 
    Written to make the code more readable
    """
    print("SELL", str(bid) + "x", bid)
    orders.append(Order(product, bid, -amount))

def get_best_sell_order():
    return

def get_best_buy_order():
    return