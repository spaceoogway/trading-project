import os

# Configuration for OrderBook
# Message types
ADD_ORDER = 'A'
EXECUTE_ORDER = 'E'
DELETE_ORDER = 'D'
# Sides
BUY_SIDE = 'B'
SELL_SIDE = 'S'
# Error messages
ORDER_ID_EXISTS_ERROR = "Order ID {order_id} already exists in the order book."
ORDER_ID_NOT_FOUND_ERROR = "Order ID {order_id} not found in the order book."
# Config for read -> process -> write order book data
NETWORK_TIME_INDEX = 0
MSG_TYPE_INDEX = 2 
SIDE_INDEX = 4
PRICE_INDEX = 5
QTY_INDEX = 7
ORDER_ID_INDEX = 8

# Configuration for OrderBookDataHandler
# Define columns to drop
OB_REMOVE_COLUMNS = ['bist_time', 'asset_name', 'que_loc']
# Define relevant message types
OB_RELEVANT_MSG_TYPES = [ADD_ORDER, EXECUTE_ORDER, DELETE_ORDER]
# Define DataFrame column names for CSV input
OB_READ_COLUMNS = ["network_time", "bist_time", "msg_type", "asset_name", "side", "price", "que_loc", "qty", "order_id"]
# Define DataFrame column names for saving data
OB_SAVE_COLUMNS = ['Date', 'Asset', 'bid3qty', 'bid3px', 'bid2qty', 'bid2px', 'bid1qty', 'bid1px',
                'ask1px', 'ask1qty', 'ask2px', 'ask2qty', 'ask3px', 'ask3qty', 'Mold Package']
# Workspace path
WORKSPACE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../../"
# Asset name
ASSET_NAME = 'AKBNK'


# Configuration for formatted strings and default file names
# String formats for prompting user input
USER_INPUT_PROMPT = "{prompt} (default: {default}): "
# String formats for timed actions
TIMED_ACTION_START = "\n{description}"
TIMED_ACTION_COMPLETE = "{action} completed in {elapsed:.3f} seconds."
LOADING_ORDERS_MSG = "Loading orders from {input_file_path}"
PROCESSING_ORDERS_MSG = "Processing orders from {input_file_path}"
SAVING_RESULTS_MSG = "Saving results to {output_file_path}"
# String formats for lob creation
LOB_SUCCESS_MSG = "\nLimiting order book created successfully in {total:.2f} seconds.\n"
# Default file names
DEFAULT_INPUT_FILEPATH = 'data/AKBNK.E.csv'
DEFAULT_LOB_FILEPATH = 'data/output/lob/AKBNK_LOB.csv'
# Argument descriptions
ARGUMENT_PARSER_DESCRIPTION = "Process order book data."
INPUT_FILE_ARGUMENT_HELP = 'Input CSV file path'
OUTPUT_FILE_ARGUMENT_HELP = 'Output CSV file path'
STREAMING_OPTION_HELP = 'Stream orders from input file to output file'

# Prediction configuration
PRED_REMOVE_COLUMNS = ['Asset', 'Mold Package']
PRED_BID_COLUMNS = ['bid3qty', 'bid3px','bid2qty', 'bid2px', 'bid1qty', 'bid1px']
PRED_ASK_COLUMNS = ['ask1px', 'ask1qty', 'ask2px', 'ask2qty', 'ask3px', 'ask3qty']