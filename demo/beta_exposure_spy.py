import pandas as pd
import yfinance as yf
from vbase_utils.stats.pit_robust_betas import calc_robust_beta

def download_daily_returns(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def get_spy_constituents():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print("Failed to fetch SPY constituents:", e)
        return ["AAPL", "MSFT", "GOOGL", "SPY"]  # fallback list

def calculate_rolling_betas(stock_returns, market_returns, window=252):
    betas = {}
    for ticker in stock_returns.columns:
        stock_series = stock_returns[ticker]
        market_series = market_returns['SPY']
        beta_series = calc_robust_beta(stock_series, market_series, window=window)
        betas[ticker] = beta_series
    return pd.DataFrame(betas)

def main():
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    print("Fetching SPY constituents...")
    tickers = get_spy_constituents()

    if 'SPY' not in tickers:
        tickers.append('SPY')

    print("Downloading return data...")
    returns = download_daily_returns(tickers[:20], start_date, end_date)  # limit to 20 tickers for speed

    print("Splitting returns...")
    market_returns = returns[['SPY']]
    stock_returns = returns.drop(columns=['SPY'])

    print("Calculating rolling betas...")
    rolling_betas = calculate_rolling_betas(stock_returns, market_returns)

    output_path = "output/rolling_betas.csv"
    rolling_betas.to_csv(output_path)
    print(f"Saved rolling betas to: {output_path}")

if __name__ == "__main__":
    # Prevent execution during CI
    import os
    if os.getenv("CI") != "true":
        main()
