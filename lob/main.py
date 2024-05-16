import os
import argparse
import time
import logging
from order_book_data_handler import OrderBookDataHandler
from order_book import OrderBook
from config.constants import (USER_INPUT_PROMPT, TIMED_ACTION_START, TIMED_ACTION_COMPLETE, LOB_SUCCESS_MSG,
                              DEFAULT_INPUT_FILEPATH, DEFAULT_LOB_FILEPATH, ARGUMENT_PARSER_DESCRIPTION,
                              INPUT_FILE_ARGUMENT_HELP, OUTPUT_FILE_ARGUMENT_HELP, LOADING_ORDERS_MSG,
                              PROCESSING_ORDERS_MSG, SAVING_RESULTS_MSG, STREAMING_OPTION_HELP)

logging.basicConfig(level=logging.INFO)


def get_user_input(prompt, default):
    """Prompt for user input, return default if no input is given."""
    formatted_prompt = USER_INPUT_PROMPT.format(prompt=prompt, default=default)
    user_input = input(formatted_prompt).strip()
    return user_input if user_input else default


def timed_action(description, function, *args, **kwargs):
    """Executes a function with timing and descriptive output."""
    print(TIMED_ACTION_START.format(description=description))
    start_time = time.time()
    result = function(*args, **kwargs)
    elapsed_time = time.time() - start_time
    print(TIMED_ACTION_COMPLETE.format(
        action=description.split(' ')[0], elapsed=elapsed_time))
    return result


def main(input_file_path, output_file_path, stream_orders_flag):
    """Main function to load, process, and save the orders."""
    main_start_time = time.time()
    order_book = OrderBook()

    if stream_orders_flag:
        print("Streaming orders...")
        # Use streaming function
        order_book.stream_orders(input_file_path, output_file_path)
    else:
        # Standard load, process, and save
        orders = timed_action(LOADING_ORDERS_MSG.format(input_file_path=input_file_path),
                              OrderBookDataHandler.load, input_file_path)

        results = timed_action(PROCESSING_ORDERS_MSG.format(input_file_path=input_file_path),
                               order_book.process_orders, orders)

        timed_action(SAVING_RESULTS_MSG.format(output_file_path=output_file_path),
                     OrderBookDataHandler.save, output_file_path, results)

    print(LOB_SUCCESS_MSG.format(total=time.time() - main_start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=ARGUMENT_PARSER_DESCRIPTION)
    parser.add_argument('--input', type=str, metavar='', help=INPUT_FILE_ARGUMENT_HELP +
                        "default value is: " + DEFAULT_INPUT_FILEPATH, default=DEFAULT_INPUT_FILEPATH)
    parser.add_argument('--output', type=str, metavar='', help=OUTPUT_FILE_ARGUMENT_HELP +
                        "default value is: " + DEFAULT_LOB_FILEPATH, default=DEFAULT_LOB_FILEPATH)
    parser.add_argument('--stream', action='store_true',
                        help=STREAMING_OPTION_HELP)

    args = parser.parse_args()

    main(args.input, args.output, args.stream)
