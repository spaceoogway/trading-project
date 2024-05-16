import pandas as pd
from typing import List, Tuple
from config.constants import OB_REMOVE_COLUMNS, OB_RELEVANT_MSG_TYPES, OB_READ_COLUMNS, OB_SAVE_COLUMNS, WORKSPACE_PATH


class OrderBookDataHandler:

    def read(file_path: str) -> pd.DataFrame:
        # Load the CSV into a DataFrame
        return pd.read_csv(WORKSPACE_PATH + file_path, names=OB_READ_COLUMNS)

    def filter(data) -> pd.DataFrame:
        # Drop the columns that are not needed
        data = data.drop(OB_REMOVE_COLUMNS, axis=1)
        # Filter the data based on the relevant message types
        data = data[data['msg_type'].isin(OB_RELEVANT_MSG_TYPES)]
        return data

    def load(file_path: str) -> List[Tuple]:
        data = OrderBookDataHandler.read(file_path)
        data = OrderBookDataHandler.filter(data)
        return data.values.tolist()

    def save(file_path: str, data: pd.DataFrame):
        data = pd.DataFrame(data, columns=OB_SAVE_COLUMNS)
        data['Date'] = pd.to_datetime(data['Date'], unit='ns')
        data.to_csv(WORKSPACE_PATH + file_path, index=False)
