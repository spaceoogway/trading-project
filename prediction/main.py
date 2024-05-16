import pandas as pd
from config.constants import PRED_REMOVE_COLUMNS, DEFAULT_LOB_FILEPATH, PRED_FILE_PARAMETERS_CAPTION, \
    PRED_DATA_FOLDER_DIR, PRED_FILE_PARAMETERS_STR, LOB_PRED_FILEPATH
import matplotlib.pyplot as plt
from functools import partial
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from millify import millify
from tqdm import tqdm
sns.set_theme()
plt.rcParams["lines.linewidth"] = 0.12
plt.rc("font", family="Helvetica")


def load_lob_data(file_path):
    lob_df = pd.read_csv(file_path)
    lob_df["Date"] = pd.to_datetime(lob_df["Date"])
    lob_df = lob_df.drop(columns=PRED_REMOVE_COLUMNS)
    return lob_df


def middle_of_liquidity_gap(row):
    if row["bid1qty"] == 0 and row["ask1qty"] == 0:
        return None
    elif row["bid1qty"] == 0:
        return row["ask1px"]
    elif row["ask1qty"] == 0:
        return row["bid1px"]
    return (row["ask1px"] + row["bid1px"]) / 2


def add_ohcl_columns(df, round_time="10s"):
    # Group the data by the rounded time and get the first and last values
    df["rounded_time"] = df["Date"].dt.floor(round_time)
    close_results = df.groupby("rounded_time")["middle"].last()
    open_results = df.groupby("rounded_time")["middle"].first()
    high_results = df.groupby("rounded_time")["middle"].max()
    low_results = df.groupby("rounded_time")["middle"].min()
    # Shift the close values by one group
    shifted_close = close_results.shift(1)
    shifted_open = open_results.shift(1)
    shifted_high = high_results.shift(1)
    shifted_low = low_results.shift(1)
    # Map these shifted group results back to the original DataFrame
    df["close"] = df["rounded_time"].map(shifted_close)
    df["open"] = df["rounded_time"].map(shifted_open)
    df["high"] = df["rounded_time"].map(shifted_high)
    df["low"] = df["rounded_time"].map(shifted_low)
    df = df.dropna(subset=["close"])
    df = df.dropna(subset=["open"])
    df = df.dropna(subset=["high"])
    df = df.dropna(subset=["low"])
    return df


def forward_fill_non_zero_values(df, col_name):
    df[col_name] = df[col_name].ffill()
    return df


def run_strategy(df, window="1000s", multiplier=3, initial_tl=100000,
                 initial_qty=0, max_order_qty=10, stop_buy_time="1h", order_delay="5min"):

    def add_bollinger_bands(df, window=window, multiplier=multiplier):
        # Calculate rolling mean and standard deviation over the window
        df["mean"] = df["close"].rolling(window=window).mean()
        df["std_dev"] = df["close"].rolling(window=window).std()

        # Calculate Bollinger Bands
        df["upper_band"] = df["mean"] + (multiplier * df["std_dev"])
        df["lower_band"] = df["mean"] - (multiplier * df["std_dev"])
        df = df[df.index > (df["rounded_time"].iloc[0] +
                            pd.Timedelta(window))].copy()
        df.dropna(subset=["upper_band"], inplace=True)
        return df

    def add_buy_sell_orders(df, stop_buy_time=stop_buy_time):
        # Initialize the buy and sell orders
        for i in range(1, 4):
            # Buy orders
            df[f"buy{i}sig"] = (df[f"ask{i}px"] < df["lower_band"]) & (
                df[f"ask{i}qty"] > 0)
            # Sell orders
            df[f"sell{i}sig"] = (df[f"bid{i}px"] > df["upper_band"]) & (
                df[f"bid{i}qty"] > 0)
        # Stop buying in the stop_buy_time time frame
        last_hour_filter = df["rounded_time"] > df["rounded_time"].max(
        ) - pd.Timedelta(stop_buy_time)
        df.loc[last_hour_filter, ["buy1sig", "buy2sig", "buy3sig"]] = False
        return df

    # Add the Bollinger Bands
    df = add_bollinger_bands(df, window=window, multiplier=multiplier)
    df.reset_index(inplace=True)
    # Initialize the current balance and quantity
    df["balance_tl"] = [initial_tl] + [None] * (len(df) - 1)
    df["balance_qty"] = [initial_qty] + [None] * (len(df) - 1)
    # Initialize the columns that saves the buy and sell orders both in TL and in quantities
    qty_change = [0] * len(df)
    tl_change = [0.0] * len(df)
    # Initialize the current balance and quantity
    current_balance_tl = initial_tl
    current_balance_qty = initial_qty
    # Filter the DataFrame to only include rows where
    # there is a buy or sell order according to Bollinger Bands
    df = add_buy_sell_orders(df)
    filtered_df = df[(df["buy1sig"] | df["sell1sig"])].copy()
    # Add order columns
    for i in range(1, 4):
        # Add order columns
        df[f"buy{i}ord"] = False
        df[f"sell{i}ord"] = False
    # Iterate over the rows
    # TODO: This is an inefficient way to do this, it can be optimized
    # Set the previous time to the first time in the filtered DataFrame
    previous_time = filtered_df["rounded_time"].iloc[0]
    for index, row in tqdm(filtered_df.iterrows()):
        # Check if the time is within the sample frequency
        if row["rounded_time"] <= previous_time + pd.Timedelta(order_delay):
            continue
        # If not, update the previous time
        previous_time = row["rounded_time"]
        # Update the maximum order quantities
        current_max_order_qty = min(max_order_qty, current_balance_qty)

        for i in range(1, 4):
            buy_key = f"buy{i}sig"
            sell_key = f"sell{i}sig"
            # Ensure buy order is true and there's enough balance to buy
            if row[buy_key]:
                ask_px = row[f"ask{i}px"]
                ask_qty = row[f"ask{i}qty"]
                current_max_order_tl = min(
                    max_order_qty*ask_px, current_balance_tl)
                # Ensure there's enough tl to buy else set buy order to false
                if current_max_order_tl < ask_px:
                    break
                else:
                    df.at[index, f"buy{i}ord"] = True
                    # Calculate the buy order quantity
                    buy_qty = min(current_max_order_tl//ask_px, ask_qty)
                    buy_tl = buy_qty*ask_px
                    # Update current balances
                    current_balance_qty += buy_qty
                    current_balance_tl -= buy_tl
                    df.at[index, "balance_tl"] = current_balance_tl
                    df.at[index, "balance_qty"] = current_balance_qty
                    # Update the qty and tl changes # TODO: inefficient
                    qty_change[index] += buy_qty
                    tl_change[index] -= buy_tl
                    # Update the maximum order qty
                    current_max_order_tl -= buy_tl
            elif row[sell_key]:
                # Ensure there's enough qty to sell else set buy order to false
                if current_max_order_qty <= 0:
                    break
                else:
                    df.at[index, f"sell{i}ord"] = True
                    bid_px = row[f"bid{i}px"]
                    bid_qty = row[f"bid{i}qty"]

                    sell_qty = min(current_max_order_qty, bid_qty)
                    sell_tl = sell_qty*bid_px
                    # Update the balances
                    current_balance_qty -= sell_qty
                    current_balance_tl += sell_tl
                    df.at[index, "balance_tl"] = current_balance_tl
                    df.at[index, "balance_qty"] = current_balance_qty
                    # Update the qty and tl changes # TODO: inefficient
                    qty_change[index] -= sell_qty
                    tl_change[index] += sell_tl
                    # Update the maximum order qty
                    current_max_order_qty -= sell_qty

    # Add the balance columns to the DataFrame
    df["qty_change"] = qty_change
    df["tl_change"] = tl_change
    # Forward fill the non-zero values
    df["balance_tl"] = df["balance_tl"].ffill()
    df["balance_qty"] = df["balance_qty"].ffill()
    return df


def create_bollinger_plot(df, strategy_values):
    # Making the x axis stretched so that the plot is more visible
    # Create a figure with 2 subplots, stacked vertically
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(13, 20), sharex=True)
    x_min = df["rounded_time"].min() - pd.Timedelta("1s")
    x_max = df["rounded_time"].max() + pd.Timedelta("1s")
    y_min = df["lower_band"].min() - 0.01*df["upper_band"].max()
    y_max = max(df["upper_band"].max(), df["ask1px"].max()) + \
        0.01*df["upper_band"].max()
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])

    # Filter the buy and sell orders
    buy_true_df = df[df["buy1ord"]]
    sell_true_df = df[df["sell1ord"]]

    # plot the buy and sell orders
    ax1.scatter(buy_true_df["rounded_time"],
                buy_true_df["ask1px"], color="green", label="buy order")
    ax1.scatter(sell_true_df["rounded_time"],
                sell_true_df["bid1px"], color="red", label="sell order")
    # plot simple moving average
    ax1.plot(df["rounded_time"], df["mean"], label="moving average", linewidth=0.3)
    # plot the bollinger bands
    ax1.plot(df["rounded_time"], df["upper_band"],
             color="blue", label="upper band", linestyle="--")
    ax1.plot(df["rounded_time"], df["lower_band"],
             color="blue", label="lower band", linestyle="--")
    # fill the area between the bands
    ax1.fill_between(df["rounded_time"], df["upper_band"],
                     df["lower_band"], color="blue", alpha=0.1)
    # plot the best bid and ask prices
    ax1.plot(df["rounded_time"], df["bid1px"], label="best bid", color="green")
    ax1.plot(df["rounded_time"], df["ask1px"], label="best ask", color="red")
    # Set up date formatting for x-axis to show only hours and minutes
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(
        interval=30))  # Adjust interval as needed
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.set_title(
        PRED_FILE_PARAMETERS_CAPTION.format(**strategy_values), fontsize=16)

    ax1.set_ylabel("Price (TL)", fontsize=14)
    # make lines in the legend thicker
    # Get the legend object
    legend = ax1.legend(loc='upper right')
    # Set the linewidth of the lines in the legend
    for line in legend.get_lines():
        line.set_linewidth(2)

    # Plot the quantity balance in the second axis of the plot
    ax2.plot(df["rounded_time"], df["balance_qty"], label="Stock Balance",
             color="darkorange", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Stock (Quantity)", fontsize=14)
    ax2.legend(loc='upper right')
    # Plot the tl balance in the third axis of the plot
    ax3.plot(df["rounded_time"], df["balance_tl"], label="Liquidity Balance",
             color="darkgreen", linewidth=1.2, alpha=0.8)
    ax3.set_ylabel("Liquidity (TL)", fontsize=14)
    # Fill the area under the line
    ax3.legend(loc='upper right')
    # Plot the balance and quantity changes in the second axis of the plot
    total_value = (df["balance_tl"] + (df["balance_qty"] * df["close"]))
    total_value_max = total_value.max()
    total_value_min = total_value.min()
    # Set the limits of the second axis
    ax4.set_xlim([x_min, x_max])
    ax4.set_ylim([total_value_min - 0.001*total_value_min,
                 total_value_max+0.001*total_value_max])

    ax4.plot(df["rounded_time"], total_value, label="Total Portfolio Value",
             color="indigo", linewidth=1, alpha=0.8)
    ax4.set_ylabel("Liquidity (TL)", fontsize=14)
    ax4.set_xlabel("Time", fontsize=14)
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(
        interval=30))  # Adjust interval as needed
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax4.legend(loc='upper right')

    # Add a text box with the initial and final balance
    # Get the initial and final balance
    initial_tl = strategy_values["initial_tl"]
    initial_qty = strategy_values["initial_qty"]
    # Create the text box
    initial_total = initial_tl + initial_qty*df["close"].iloc[0]
    # Get the final balance

    final_balance = df["balance_tl"].iloc[-1]
    final_balance_qty = df["balance_qty"].iloc[-1]
    p_millify = partial(millify, precision=2)

    # Stock Information
    text_stock = f"Initial Stock Amount (QTY): {p_millify(int(initial_qty))}\n" \
                f"Initial Stock Value (TL): {p_millify(int(initial_qty*df['close'].iloc[0]))} TL\n" \
                f"Final Stock Amount (QTY): {p_millify(int(final_balance_qty))}\n" \
                f"Final Stock Value (TL): {p_millify(int(final_balance_qty*df['close'].iloc[-1]))} TL"


    # Balance Information
    text_balance = f"Initial Balance (TL): {p_millify(int(initial_tl))} TL\n" \
                f"Final Balance (TL): {p_millify(int(final_balance))} TL"

    # Profit Information
    text_profit = f"Profit (TL): {p_millify(int(final_balance + final_balance_qty*df['close'].iloc[-1] - initial_total))} TL\n" \
              f"Profit (%): {((final_balance + final_balance_qty*df['close'].iloc[-1] - initial_total)/initial_total)*100:.2f}%"


    # Portfolio Value Information
    text_portfolio = f"Initial Portfolio Value (TL): {p_millify(int(initial_total))} TL\n" \
                     f"Final Portfolio Value (TL): {p_millify(int(final_balance + final_balance_qty*df['close'].iloc[-1]))} TL"

# You can now place these text boxes in appropriate subplots as discussed earlier.
    props = dict(boxstyle="round", facecolor="white", alpha=0.6)
    # Place text box in upper left in axes coords
    ax2.text(0.02, 0.04, text_stock, transform=ax2.transAxes, fontsize=10,
             verticalalignment="bottom", bbox=props, linespacing=1.5)
    ax3.text(0.02, 0.04, text_balance, transform=ax3.transAxes, fontsize=10,
             verticalalignment="bottom", bbox=props, linespacing=1.5)
    ax4.text(0.02, 0.04, text_profit + "\n" + text_portfolio, transform=ax4.transAxes, fontsize=10,
             verticalalignment="bottom", bbox=props, linespacing=1.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # save the plot with high resolution
    # use millify to convert the balance to a more readable format (e.g. 1M instead of 1000000)
    strategy_values["initial_tl"] = millify(strategy_values["initial_tl"])
    strategy_values["initial_qty"] = millify(strategy_values["initial_qty"])
    plt.savefig(PRED_DATA_FOLDER_DIR +
                f"plots/bollinger_plot" + PRED_FILE_PARAMETERS_STR.format(**strategy_values) + ".png", dpi=300)


def get_buy_sell_df(df):
    return df[(df["lower_band"] < df["ask1px"]) | (df["upper_band"] > df["bid1px"])]


def main():
    #  Filter and add columns to the data
    file_path = DEFAULT_LOB_FILEPATH
    # Load the data
    print("Loading the data...")
    lob_df = load_lob_data(file_path)
    # Add the middle point of liquidity gap to use as the representative price value
    # to use it for Bollinger Bands.
    lob_df["middle"] = lob_df.apply(middle_of_liquidity_gap, axis=1)
    lob_df = lob_df.dropna(subset=["middle"])
    # Add the OHCL columns
    lob_df = add_ohcl_columns(lob_df)
    # Set the Date column as the index
    lob_df.set_index("Date", inplace=True)
    # TODO: there is an odd spike at the end of the data, remove it but it's not a good solution
    lob_df = lob_df[:-500].copy()
    # Create the bid and ask DataFrames
    strategy_values = {
        "window": "1h",
        "multiplier": 2,
        "initial_tl": 100000,
        "initial_qty": 10000,
        "max_order_qty": 5000,
        "order_delay": "1min",
        "stop_buy_time": "10min"
    }
    # Run the strategy
    print("Running the strategy...")
    results_df = run_strategy(lob_df, **strategy_values)
    # Create the Bollinger plot
    print("Creating the Bollinger plot...")
    create_bollinger_plot(results_df, strategy_values)

    # Save the results
    print("Saving the results...")
    results_df.to_csv(LOB_PRED_FILEPATH + PRED_FILE_PARAMETERS_STR.format(**
                      strategy_values) + ".csv", index=False)


if __name__ == "__main__":
    main()
