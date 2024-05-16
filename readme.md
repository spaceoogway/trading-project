
# Trading Project

This project consists of two main parts: preparing a limit order book (LOB) and making a trading strategy based on Bollinger Bands. The instructions below explain how to use the code.

## Prerequisites

- Python 3 (tested with Python 3.11.6)

```sh
pip install -r requirements.txt
```

## Usage

### Preparing the Limit Order Book

To run the script that prepares the limit order book, execute the following command:

```sh
python3 lob/main.py
```

By default, this command reads the file `data/AKBNK.E.csv` and creates the file `data/output/lob/AKBNK_LOB.csv`.

#### Changing Input and Output Files

You can specify different input and output files using command-line arguments:

```sh
python3 lob/main.py --input <input_file_path> --output <output_file_path>
```

For example, to change the output file name:

```sh
python3 lob/main.py --output data/output/lob/AKBNK_LOB_2.csv
```

Or to change the input file:

```sh
python3 lob/main.py --input data/AKBNK.e2.csv
```

#### Using Streaming Mode

If you want to create the limit order book in streaming mode (useful in case of memory restrictions and may work faster for small files; tested for 20MB with 230k rows, 9 columns), use the `--stream` flag:

```sh
python3 lob/main.py --stream
```

For more information about the commands:

```sh
python3 lob/main.py -h
```

### Running the Prediction Module

To run the prediction module, use the following command:

```sh
python3 prediction/main.py
```

The prediction module expects the file `data/lob/AKBNK_LOB.csv` to be present. Ensure that you have generated this file using the `python3 lob/main.py` command or have placed it in the appropriate directory before running the prediction module.

## Code Overview

### Limit Order Book Preparation (`lob/main.py`)

The `lob/main.py` script handles loading, processing, and saving orders from a CSV file. The key components include:

- **OrderBookDataHandler**: A class responsible for reading, filtering, and saving order data.
- **OrderBook**: A class representing the limit order book, which manages adding, executing, and deleting orders.
- **Main Function**: Parses command-line arguments and coordinates the process of loading, processing, and saving orders.

### Prediction Module (`prediction/main.py`)

The `prediction/main.py` script implements a trading strategy based on Bollinger Bands. It performs the following steps:

1. Loads the limit order book data.
2. Calculates the Bollinger Bands.
3. Generates buy and sell signals based on the strategy.
4. Gives buy and sell orders based on the available amount of stocks, user's holdings and given parameters.
5. Plots the results and saves the final data to a CSV file.

### Prediction Module Parameters

The prediction module uses the following parameters in the main.py file to run the Bollinger Bands strategy:

```python
strategy_values = {
    "window": "1h",          # Time window for Bollinger Bands calculation
    "multiplier": 2,         # Multiplier for the standard deviation in Bollinger Bands
    "initial_tl": 100000,    # Initial liquidity balance in TL
    "initial_qty": 10000,    # Initial quantity of the stock
    "max_order_qty": 5000,   # Maximum quantity for each order
    "order_delay": "1min",   # Minimum delay between orders
    "stop_buy_time": "10min" # Time before the end of the session to stop placing buy orders
}
```

- `window`: The time window used for calculating the Bollinger Bands.
- `multiplier`: The multiplier applied to the standard deviation to calculate the upper and lower bands.
- `initial_tl`: The initial liquidity balance in Turkish Lira (TL).
- `initial_qty`: The initial quantity of the stock.
- `max_order_qty`: The maximum quantity for each buy or sell order.
- `order_delay`: The minimum delay between consecutive orders to spread trading on various signals.
- `stop_buy_time`: The time period before the end of the trading session during which no new buy orders will be placed.

## Contact

If you have any questions please reach me at [akyuz.kn@gmail.com].
