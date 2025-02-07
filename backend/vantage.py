import requests
import json
import os
import pandas as pd
import ta  # Technical Analysis library
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Replace with your own free tier API key from Alpha Vantage
API_KEY = os.getenv("Alpha_Vantage_KEY")

def get_global_quote(symbol):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("Global Quote", {})
    else:
        print("Error fetching GLOBAL_QUOTE:", response.status_code)
        return {}

def get_daily_time_series(symbol):
    """
    Fetch historical daily price data for a given symbol and return it as a pandas DataFrame.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=compact"
    response = requests.get(url)
    if response.status_code == 200:
        time_series = response.json().get("Time Series (Daily)", {})
        if not time_series:
            return None
        # Create DataFrame from the time series dictionary
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        }, inplace=True)
        # Convert index to datetime and data to float
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        # Sort by date ascending
        df.sort_index(inplace=True)
        return df
    else:
        print("Error fetching TIME_SERIES_DAILY:", response.status_code)
        return None

def compute_macd(symbol):
    """
    Computes the MACD and signal line using the close prices from daily data.
    Returns a dictionary with the latest MACD and Signal values.
    """
    df = get_daily_time_series(symbol)
    if df is None or df.empty:
        return {}
    # Initialize MACD indicator with typical parameters
    macd_indicator = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()
    if macd_line.empty or signal_line.empty:
        return {}
    # Get the most recent (latest) values
    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    return {"MACD": round(latest_macd, 4), "Signal": round(latest_signal, 4)}

def get_rsi(symbol, time_period=14):
    url = f"https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval=daily&time_period={time_period}&series_type=close&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        rsi_data = data.get("Technical Analysis: RSI", {})
        if rsi_data:
            latest_date = sorted(rsi_data.keys(), reverse=True)[0]
            return rsi_data[latest_date]
        else:
            return {}
    else:
        print("Error fetching RSI:", response.status_code)
        return {}

def combined_analysis(symbol):
    quote = get_global_quote(symbol)
    # Compute MACD locally
    macd = compute_macd(symbol)
    rsi = get_rsi(symbol)
    return {
        "Global Quote": quote,
        "MACD": macd,
        "RSI": rsi
    }

def display_analysis(symbol, analysis):
    print(f"\nCombined Analysis for {symbol}")
    print("=" * 50)

    # Global Quote Section
    global_quote = analysis.get("Global Quote", {})
    if global_quote:
        print("\nGlobal Quote:")
        for key, value in global_quote.items():
            print(f"  {key}: {value}")
    else:
        print("\nGlobal Quote: No data available.")

    # MACD Section (locally computed)
    macd = analysis.get("MACD", {})
    if macd:
        for key, value in macd.items():
            print(f"  {key}: {value}")
    else:
        print("\nMACD: No data available.")

    # RSI Section
    rsi = analysis.get("RSI", {})
    if rsi:
        # If RSI dictionary only contains one key "RSI", print it on one line.
        if list(rsi.keys()) == ["RSI"]:
            print(f"\nRSI: {rsi['RSI']}")
        else:
            print("\nRSI:")
            for key, value in rsi.items():
                print(f"\n{key}: {value}")
    else:
        print("\nRSI: No data available.")

if __name__ == "__main__":
    print("Which stock would you like to view? (US Market Only)")
    symbol = input().strip().upper()  # Ensure symbol is uppercase
    analysis = combined_analysis(symbol)
    display_analysis(symbol, analysis)

