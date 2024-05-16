import logging
from typing import Dict, List, Tuple
import numpy as np
from sortedcontainers import SortedDict
from tqdm import tqdm

from config.constants import (ADD_ORDER, BUY_SIDE, DELETE_ORDER, EXECUTE_ORDER,
                    ORDER_ID_EXISTS_ERROR, ORDER_ID_NOT_FOUND_ERROR, SELL_SIDE, ASSET_NAME)


class OrderBook:
    def __init__(self):
        self._price_levels_dict: Dict[str, Dict[float, int]] \
            = {BUY_SIDE: SortedDict(), SELL_SIDE: SortedDict()}
        self._orders_dict: Dict[int, Tuple[float, int]] = {}
        self.mold_package = []
        logging.basicConfig(level=logging.INFO)

    def add_order(self, side: str, price: float, qty: int, order_id: int) -> None:
        """Add a new order."""
        self._price_levels_dict[side][price] = self._price_levels_dict[side].get(
            price, 0) + qty
        self._orders_dict[order_id] = (price, qty)

    def execute_order(self, side: str, order_id: int, qty: int) -> None:
        """Execute an order."""
        # Unpack the order and update the order quantity
        price, existing_qty = self._orders_dict[order_id]
        new_order_qty = existing_qty - qty
        if new_order_qty <= 0:
            del self._orders_dict[order_id]
        else:
            self._orders_dict[order_id] = (price, new_order_qty)
        # Update the price levels
        new_price_qty = self._price_levels_dict[side][price] - qty
        if new_price_qty <= 0:
            del self._price_levels_dict[side][price]
        else:
            self._price_levels_dict[side][price] = new_price_qty
        return price

    def delete_order(self, side: str, order_id: int) -> None:
        """Delete an order."""
        # Unpack the order and delete the order
        price, qty = self._orders_dict.pop(order_id)
        # Update the price quantity and delete the price if necessary
        self._price_levels_dict[side][price] -= qty
        if self._price_levels_dict[side][price] <= 0:
            del self._price_levels_dict[side][price]
        return price, qty

    def get_order_book_state(self, current_time: int) -> Tuple:
        """Get the limit order book state."""
        # Get the best bid and ask prices
        best_bids = tuple(self._price_levels_dict[BUY_SIDE].items()[-3:])
        best_asks = tuple(self._price_levels_dict[SELL_SIDE].items()[:3])
        # Pad best bid
        if len(best_bids) < 3:
            best_bids = ((0, 0),) * (3 - len(best_bids)) + best_bids
        # Pad top ask
        if len(best_asks) < 3:
            best_asks = best_asks + ((0, 0),) * (3 - len(best_asks))
        # Reverse the best bid pairs to display quantity first and price second
        best_bid_pairs = [tuple(reversed(pair)) for pair in best_bids]
        # Flatten the best bid and ask prices
        best_bids = [element for pair in best_bid_pairs for element in pair]
        best_asks = [element for pair in best_asks for element in pair]
        # Create the order book state
        # TODO: Get the asset name from the orders
        order_book_state = (current_time, ASSET_NAME, *best_bids, *
                            best_asks, ';'.join(self.mold_package))
        # Reset the mold package
        self.mold_package = []
        # Return the order book state
        return order_book_state

    def check_order_id_in_order_book(self, order_id: int, msg_type: str) -> None:
        """Check if the order ID is in the order book and raise an error if necessary."""
        # If the order ID is in the book and the order is ADD raise an error
        if msg_type == ADD_ORDER and order_id in self._orders_dict:
            raise ValueError(ORDER_ID_EXISTS_ERROR.format(order_id=order_id))

        # If the order ID is not in the book and the msg_type is DELETE or EXECUTE raise an error
        elif (msg_type == EXECUTE_ORDER or msg_type == DELETE_ORDER) and order_id not in self._orders_dict:
            raise ValueError(
                ORDER_ID_NOT_FOUND_ERROR.format(order_id=order_id))

    def process_order(self, order) -> List[Tuple]:
        """Process order and update the limit order book."""
        # Unpack the order
        msg_type, side, price, qty, order_id = order
        # Check if the order ID is and raise an error if necessary
        self.check_order_id_in_order_book(order_id, msg_type)
        # Process the order based on the message type
        if msg_type == ADD_ORDER:
            self.add_order(side, price, qty, order_id)
        elif msg_type == EXECUTE_ORDER:
            price = self.execute_order(side, order_id, qty)
        elif msg_type == DELETE_ORDER:
            price, qty = self.delete_order(side, order_id)
        # return the mold
        self.mold_package.append(f"{msg_type}-{side}-{price}-{qty}-{order_id}")

    def process_orders(self, orders: List[Tuple]) -> List[Tuple]:
        """Process all orders iteratively and return the resulting limit order book."""
        # Initialize the results and mold package lists, and the current time
        results = []
        # Get the network time of the first order
        current_time = orders[0][0]
        # Iterate over the orders
        for order in tqdm(orders):
            # Get the network time of the order
            network_time = order[0]
            # If the current time is different from the network time, get the order book state.
            if current_time != network_time:
                # Get the best bid & ask levels and the mold package and append the result
                results.append(self.get_order_book_state(current_time))
                # Reset the mold package and update the current time
                current_time = network_time
            # Process the order and append the mold to the mold package
            self.process_order(order[1:])
        # Append the last order book state
        results.append(self.get_order_book_state(network_time))
        return results

    def stream_orders(self, input_file_path: str, output_file_path: str) -> None:
        """Read orders, process them, and write the results iteratively to the limit order book CSV file."""
        import csv
        from config.constants import (MSG_TYPE_INDEX, NETWORK_TIME_INDEX, ORDER_ID_INDEX,
                            PRICE_INDEX, QTY_INDEX, OB_RELEVANT_MSG_TYPES,
                            OB_SAVE_COLUMNS, SIDE_INDEX)

        # Define a function to format a row into an order
        def format_row_to_order(row: Tuple) -> Tuple:
            return int(row[NETWORK_TIME_INDEX]), (
                int(row[NETWORK_TIME_INDEX]),
                row[MSG_TYPE_INDEX],
                row[SIDE_INDEX],
                float(row[PRICE_INDEX]),
                int(row[QTY_INDEX]),
                int(row[ORDER_ID_INDEX])
            )
        # Open the input file
        with open(input_file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[MSG_TYPE_INDEX] not in OB_RELEVANT_MSG_TYPES:
                    continue  # Skip irrelevant message types
                else:
                    # Get the first order and process it
                    current_time, first_order = format_row_to_order(row)
                    self.process_order(first_order[1:])
                    break
            # Open the output file
            with open(output_file_path, 'w') as out_file:
                # Create a CSV writer
                writer = csv.writer(out_file)
                # Write the header
                writer.writerow(OB_SAVE_COLUMNS)
                # Iterate over the orders
                for row in reader:
                    if row[MSG_TYPE_INDEX] not in OB_RELEVANT_MSG_TYPES:
                        continue  # Skip irrelevant message types
                    network_time, order = format_row_to_order(row)
                    # If the current time is different from the network time, get the order book state.
                    if network_time != current_time:
                        # Write the previous time and values before resetting
                        if current_time is not None:
                            order_state = self.get_order_book_state(
                                current_time)
                            writer.writerow(
                                (np.datetime64(order_state[0], 'ns'), *order_state[1:]))
                        current_time = network_time
                    self.process_order(order[1:])
                # Write the last order book state
                writer.writerow(
                    (np.datetime64(order_state[0], 'ns'), *order_state[1:]))