import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Replace with your own free tier API key from Alpha Vantage
API_KEY = os.getenv("Alpha_Vantage_KEY")

def get_exchange_rate(from_currency, to_currency):
    """
    Fetches the real-time currency exchange rate from Alpha Vantage.
    
    Args:
        from_currency (str): The source currency code (e.g., 'USD').
        to_currency (str): The target currency code (e.g., 'EUR').
    
    Returns:
        dict: A dictionary with the exchange rate information, or an empty dict if an error occurs.
    """
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}"
        f"&to_currency={to_currency}&apikey={API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("Realtime Currency Exchange Rate", {})
    else:
        print("Error fetching exchange rate:", response.status_code)
        return {}

def format_exchange_info(exchange_data, from_currency, to_currency):
    """
    Builds a formatted string with the exchange data
    (bid, ask, rate, etc.). Returns that string.
    """
    if not exchange_data:
        return "No data available for the specified currency pair."

    rate = exchange_data.get('5. Exchange Rate', 'N/A')
    last_refreshed = exchange_data.get('6. Last Refreshed', 'N/A')

    lines = []
    lines.append(f"Exchange Rate from {from_currency} to {to_currency}:")
    lines.append(f"  Exchange Rate: {rate}")
    lines.append(f"  Last Refreshed: {last_refreshed}")

    return "\n".join(lines)