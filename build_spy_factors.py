import pandas as pd
import numpy as np
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_sp500_tickers_and_sectors():
    """Fetch S&P 500 tickers and their GICS sectors from Wikipedia."""
    wiki_df = pd.read_html(WIKI_URL)[0]
    tickers = wiki_df["Symbol"].tolist()
    sectors = wiki_df.set_index("Symbol")["GICS Sector"]
    return tickers, sectors

def sanitize_tickers(tickers):
    """Replace dots with dashes for yfinance compatibility."""
    return [t.replace(".", "-") for t in tickers]

def download_adjusted_prices(tickers, start="2018-01-01", end="2025-06-10"):
    """Download adjusted close prices for given tickers from Yahoo Finance."""
    raw = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=True)
    close_prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers}, index=raw.index)
    return close_prices

def compute_daily_returns(prices):
    """Compute daily percentage returns."""
    return prices.pct_change().dropna(how="all")

def compute_betas(returns, market_col="SPY", tickers=None):
    """Compute OLS betas of each stock against the market (SPY)."""
    if market_col not in returns.columns:
        raise KeyError(f"{market_col} is missing from returns.")
    market_var = returns[market_col].var()
    cov = returns.cov().loc[tickers, market_col]
    return (cov / market_var).rename("beta")

def build_factor_panel(start="2018-01-01", end="2025-06-10"):
    """Construct the full factor panel with beta and sector dummies."""
    tickers_orig, sector_series = fetch_sp500_tickers_and_sectors()
    tickers_yf = sanitize_tickers(tickers_orig)
    all_tickers = tickers_yf + ["SPY"]

    prices = download_adjusted_prices(all_tickers, start, end)
    returns = compute_daily_returns(prices)

    betas = compute_betas(returns, market_col="SPY", tickers=tickers_yf)
    betas.index = tickers_orig  # restore original tickers with dots

    sector_dummies = pd.get_dummies(sector_series, prefix="sector")

    factor_panel = pd.concat([betas, sector_dummies], axis=1)
    return factor_panel

if __name__ == "__main__":
    panel = build_factor_panel()
    print("Done. Panel shape:", panel.shape)
    print(panel.head(10))
