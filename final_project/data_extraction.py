import yfinance as yf
import pandas as pd
from datetime import datetime
from nasdaqtrader import NasdaqTrader

def fetch_stock_data(symbol, start_date, end_date):
    # Fetch historical stock prices
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate 200-day and 35-day moving averages
    stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['35_MA'] = stock_data['Close'].rolling(window=35).mean()

    # Fetch financial data
    info = yf.Ticker(symbol).info
    financial_data = {
        'Symbol': symbol,
        'Revenues': info.get('revenueTTM', None),
        'Earnings': info.get('earningsTTM', None),
        'FutureGrowth': info.get('fiveYearAvgDividendYield', None),
        'ReturnOnEquity': info.get('returnOnEquity', None),
        'ProfitMargins': info.get('profitMargins', None),
    }

    # Convert financial data to DataFrame
    financial_df = pd.DataFrame([financial_data])

    return stock_data, financial_df

# Example for a single company (AAPL)
symbol = 'AAPL'
start_date = '2013-01-01'
end_date = '2023-01-01'

stock_prices, financial_metrics = fetch_stock_data(symbol, start_date, end_date)

# Display the DataFrames
print("Stock Prices:")
print(stock_prices.head())

print("\nFinancial Metrics:")
print(financial_metrics)


# get top 50 companies of the nasdaq
def get_top_nasdaq_companies(date, top_n=50):
    # Initialize NasdaqTrader
    nasdaq = NasdaqTrader()

    # Get the list of top companies on NASDAQ at a specific date
    top_nasdaq_companies = nasdaq.get_top_nasdaq_companies(date, top_n)

    return top_nasdaq_companies

def get_ticker_cusip_mapping(companies):
    # Fetch additional details including CUSIP using the `yfinance` package
    cusip_mapping = {}

    for symbol in companies['Symbol']:
        try:
            company = yf.Ticker(symbol)
            info = company.info
            cusip = info.get('cusip', None)

            cusip_mapping[symbol] = cusip
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    return cusip_mapping

# Example: Get top 50 NASDAQ companies on a specific date
target_date = datetime(2023, 1, 1)
top_nasdaq_companies = get_top_nasdaq_companies(target_date)

# Display the list of companies
print("Top 50 NASDAQ Companies:")
print(top_nasdaq_companies[['Symbol', 'Company Name']])

# Get ticker-CUSIP mapping
cusip_mapping = get_ticker_cusip_mapping(top_nasdaq_companies)

# Display the Ticker-CUSIP mapping
print("\nTicker-CUSIP Mapping:")
for symbol, cusip in cusip_mapping.items():
    print(f"{symbol}: {cusip}")