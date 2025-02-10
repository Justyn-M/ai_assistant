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

def display_exchange_rate(exchange_data, from_currency, to_currency):
    """
    Displays a formatted summary of the exchange rate data.
    
    Args:
        exchange_data (dict): The exchange rate information.
        from_currency (str): The source currency code.
        to_currency (str): The target currency code.
    """
    if exchange_data:
        print(f"\nExchange Rate from {from_currency} to {to_currency}:")
        print(f"  Exchange Rate: {exchange_data.get('5. Exchange Rate', 'N/A')}")
        print(f"  Last Refreshed: {exchange_data.get('6. Last Refreshed', 'N/A')}")
        print(f"  Bid Price: {exchange_data.get('8. Bid Price', 'N/A')}")
        print(f"  Ask Price: {exchange_data.get('9. Ask Price', 'N/A')}")
    else:
        print("No data available for the specified currency pair.")

if __name__ == "__main__":
    print("Currency Conversion")
    from_curr = input("Enter source currency (e.g., USD): ").strip().upper()
    to_curr = input("Enter target currency (e.g., EUR): ").strip().upper()
    
    exchange_data = get_exchange_rate(from_curr, to_curr)
    display_exchange_rate(exchange_data, from_curr, to_curr)
