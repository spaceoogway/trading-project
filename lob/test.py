import pandas as pd
from order_book_data_handler import OrderBookDataHandler
import time
from tqdm import tqdm
input_file_name = 'data/AKBNK.E.csv'
orders = OrderBookDataHandler.load(input_file_name)
orders_df = pd.DataFrame(orders, columns=["network_time", "msg_type", "side", "price", "qty", "order_id"])

# Performance of turning a DataFrame into a list
now = time.time()
rows = orders_df.values.tolist()
current_time = rows[0][0]

for row in rows:
    network_time = row[0]
    if current_time != network_time:
        current_time = network_time
    else:
        pass

print(f"Processed list in {time.time() - now:.4f} seconds")

# Performance of itertuples, somehow this is sometimes faster than the list approach, 
# sometimes slower. I will use list approach.
now = time.time()
current_time = orders_df.iloc[0]['network_time']

for order in orders_df.itertuples(index=False):
    network_time = order[0]
    if current_time != network_time:
        current_time = network_time
    else:
        pass
    pass

print(f"Processed itertuples in {time.time() - now:.4f} seconds")

# Performance of groupby and itertuples
now = time.time()
grouped = orders_df.groupby('network_time')
for group_name, group_data in tqdm(grouped):
    for row in group_data.itertuples(index=False):
        # Do something with each row in the group
        pass
print(f"Processed groupby data in {time.time() - now:.4f} seconds")


