import pandas as pd
import yfinance as yf
import os
import time

def fetch_stock_data(symbol, year):
    try:
        ticker = yf.Ticker(symbol)
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        hist = ticker.history(start=start, end=end)
        if hist.empty:
            print(f"  ⚠️ No data for {symbol}")
            return None
        hist.reset_index(inplace=True)
        hist['Stock'] = symbol
        hist['Year'] = year
        return hist[['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    except Exception as e:
        print(f"  ❌ Error for {symbol}: {e}")
        return None

def main():
    year = 2020
    input_csv = 'nse_data.csv'
    output_csv = f'data/nse_data_{year}.csv'
    os.makedirs("data", exist_ok=True)

    df_symbols = pd.read_csv(input_csv)
    symbols = df_symbols['Symbol'].astype(str).str.strip().tolist()
    yf_symbols = [sym + ".NS" for sym in symbols]

    # Track already processed stocks
    processed = set()
    if os.path.exists(output_csv):
        try:
            processed_df = pd.read_csv(output_csv, usecols=['Stock'])
            processed = set(processed_df['Stock'].unique())
        except Exception:
            pass

    print(f"📊 Starting year {year} | Skipping {len(processed)} already done.")

    batch = []
    total_rows_written = 0

    for i, symbol in enumerate(yf_symbols, 1):
        if symbol in processed:
            continue

        print(f"[{i}/{len(yf_symbols)}] Downloading: {symbol}")
        data = fetch_stock_data(symbol, year)
        if data is not None:
            batch.append(data)

        # Save every 100 stocks or at the very end
        if len(batch) >= 100 or i == len(yf_symbols):
            combined_df = pd.concat(batch, ignore_index=True)
            combined_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            total_rows_written += len(combined_df)
            print(f"✅ Wrote {len(combined_df)} rows (Total so far: {total_rows_written})")
            batch.clear()

        time.sleep(1)  # Respect API limits

    print(f"\n✅✅ Finished. Total rows written to CSV: {total_rows_written}")

if __name__ == "__main__":
    main()
