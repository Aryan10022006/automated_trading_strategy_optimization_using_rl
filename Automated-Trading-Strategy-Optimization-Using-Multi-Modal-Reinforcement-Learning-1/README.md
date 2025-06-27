Stock Data Collection and Analysis Scripts
This repository contains two Python scripts designed to help you work with historical stock data: one for collecting the data and another for performing machine learning analysis on it.

1. data_collection.py - Stock Data Collection Script
This script is designed to fetch historical stock data for multiple Indian (NSE) stocks using the yfinance library and save it into a single CSV file. It helps in automating the process of gathering financial data for further analysis.

Features (data_collection.py)
Bulk Data Download: Fetches data for a list of stock symbols from a specified input CSV.

Year-Specific Data: Downloads historical data for a specific year.

Resume Capability: Skips stocks that have already been successfully downloaded and saved to the output CSV, allowing you to resume interrupted downloads.

Rate Limiting: Includes a small delay between requests to respect API rate limits.

Organized Output: Saves the collected data into a dedicated data/ directory.

Setup (data_collection.py)
Before running the script, ensure you have Python installed. Then, install the necessary libraries using pip:

pip install pandas yfinance

Usage (data_collection.py)
Prepare your input CSV: Create a CSV file named nse_data.csv (or whatever you set input_csv to in the main() function) in the same directory as your script. This file should contain a column named Symbol listing the NSE stock symbols you wish to download (e.g., RELIANCE, TCS, INFY).

nse_data.csv example:

Symbol
RELIANCE
TCS
INFY
HDFCBANK

Run the script: Execute the Python script from your terminal:

python data_collection.py

The script will print its progress, indicating which symbols it is downloading and how many rows have been written to the output file.

Input File Format (data_collection.py)
The script expects an input CSV file (default: nse_data.csv) with at least one column:

Symbol: Contains the base ticker symbol for the Indian stock (e.g., RELIANCE, TCS). The script automatically appends .NS to convert them into yfinance compatible symbols.

Output File Format (data_collection.py)
The collected data will be saved in a CSV file within a data/ directory (e.g., data/nse_data_2020.csv). The output CSV will have the following columns:

Date: The date of the stock data (YYYY-MM-DD).

Stock: The yfinance compatible stock symbol (e.g., RELIANCE.NS).

Open: The opening price of the stock.

High: The highest price of the stock during the day.

Low: The lowest price of the stock during the day.

Close: The closing price of the stock.

Volume: The trading volume for the day.

Dividends: Dividend amount (if any).

Stock Splits: Stock split information (if any).

Year: The year for which the data was fetched.

Important Notes (data_collection.py)
Year Configuration: The year variable inside the main() function is set to 2020 by default. If you need data for a different year, modify this variable in the script.

API Limits: The script includes a time.sleep(1) call to pause for 1 second between each stock download. This is important to avoid hitting yfinance (or Yahoo Finance) API rate limits, which could lead to temporary bans or errors.

Error Handling: The script includes basic error handling for individual stock downloads. If data for a specific symbol cannot be fetched, it will print a warning and continue with the next symbol.

Large Data Sets: For a very large number of symbols or multiple years, consider running the script in batches or over several sessions, leveraging the resume capability.

2. stock_analysis.py - Stock Price Prediction and Classification Script
This script performs various machine learning tasks on historical stock data to predict future prices and classify price movements. It utilizes your collected data (e.g., from data_collection.py) for analysis.

Features (stock_analysis.py)
Data Preprocessing: Loads stock data from a CSV, converts date columns, filters by stock symbol, and handles missing values.

Linear Regression: Predicts the next day's closing price.

Calculates and prints Mean Squared Error (MSE).

Plots actual vs. predicted closing prices.

Logistic Regression: Classifies whether the stock price will go up or down on the next day.

Calculates and prints accuracy score.

Generates and plots a confusion matrix heatmap.

K-Nearest Neighbors (KNN) Classification: Classifies stock price movement for varying k values.

Evaluates and prints accuracy for k=3, 5, 7.

Setup (stock_analysis.py)
Ensure you have Python installed. Then, install the necessary libraries using pip:

pip install pandas numpy scikit-learn matplotlib seaborn

Usage (stock_analysis.py)
Prepare your input data: Ensure you have a CSV file named stock_data.csv (or specify the correct path in the CSV_FILE_PATH variable within the script). This file should contain the historical stock data in the format provided by yfinance or the data_collection.py script. If you used data_collection.py, you might need to manually copy/rename one of the output files (e.g., data/nse_data_2020.csv) to stock_data.csv in the same directory as stock_analysis.py for it to be found automatically.

The script expects columns like Date, Stock, Open, High, Low, Close, Volume, Dividends, and Stock Splits.

Configure Stock to Analyze: By default, the script processes data for 'RELIANCE.'. If your stock_data.csv contains data for multiple stocks and you wish to analyze a different one, modify the ticker_to_process variable in stock_analysis.py to match the exact ticker symbol (e.g., 'TCS.NS').

Run the script: Execute the Python script from your terminal:

python stock_analysis.py

The script will print the performance metrics for each model and display plots for Linear Regression predictions and Logistic Regression's confusion matrix.

Input File Format (stock_analysis.py)
The script expects a CSV file (default: stock_data.csv) with historical stock data, typically generated by yfinance or the data_collection.py script. Key columns used for analysis include:

Date

Stock (if multiple stocks are present in the CSV)

Open

High

Low

Close

Volume

Important Notes (stock_analysis.py)
File Dependency: This script requires the stock_data.csv file to be present. It will raise an error and exit if the file is not found.

Time-Series Split: The data splitting for training and testing preserves the chronological order (shuffle=False), which is essential for time-series analysis.

Single Stock Analysis: As configured, the script analyzes one stock at a time. If your input CSV has multiple stocks, it will filter down to the one specified in ticker_to_process.