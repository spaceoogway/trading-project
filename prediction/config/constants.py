# Prediction configuration
import os
WORKSPACE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../../"

DEFAULT_LOB_FILEPATH = WORKSPACE_PATH + 'data/output/lob/AKBNK_LOB.csv'
LOB_PRED_FILEPATH = WORKSPACE_PATH + 'data/output/prediction/AKBNK_LOB_PRED'
PRED_DATA_FOLDER_DIR = WORKSPACE_PATH + 'data/output/prediction/'
PRED_FILE_PARAMETERS_STR = "-{window}-{multiplier}-{initial_tl}-{initial_qty}-{max_order_qty}-{order_delay}-{stop_buy_time}"
PRED_FILE_PARAMETERS_CAPTION = "window={window}, multiplier={multiplier}, maximum order quantity={max_order_qty}, order delay={order_delay}"
PRED_REMOVE_COLUMNS = ['Asset', 'Mold Package']
PRED_BID_COLUMNS = ['bid3qty', 'bid3px','bid2qty', 'bid2px', 'bid1qty', 'bid1px']
PRED_ASK_COLUMNS = ['ask1px', 'ask1qty', 'ask2px', 'ask2qty', 'ask3px', 'ask3qty']
