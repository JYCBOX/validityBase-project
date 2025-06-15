import pandas as pd
import numpy as np
import sys

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_sp500_tickers_and_sectors():
    """Fetch S&P 500 tickers and their GICS sectors from Wikipedia."""
    wiki_df = pd.read_html(WIKI_URL)[0]
    tickers = wiki_df["Symbol"].tolist()
    sectors = wiki_df.set_index("Symbol")["GICS Sector"]
    return tickers, sectors

def sanitize_tickers(tickers):
    """Replace dots with dashes for file/column compatibility."""
    return [t.replace(".", "-") for t in tickers]

def load_prices(filepath):
    """
    Load adjusted close prices from a CSV file.
    Expects: rows = dates (index), columns = tickers (SPY + S&P 500)
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

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

def build_factor_panel(prices_df):
    """Build factor panel with trailing beta to SPY and sector dummies."""
    tickers_orig, sector_series = fetch_sp500_tickers_and_sectors()
    tickers_sanitized = sanitize_tickers(tickers_orig)
    all_expected = tickers_sanitized + ["SPY"]

    # Verify tickers exist in data
    missing = [t for t in all_expected if t not in prices_df.columns]
    if missing:
        raise ValueError(f"Missing tickers in price data: {missing}")

    returns = compute_daily_returns(prices_df)
    betas = compute_betas(returns, market_col="SPY", tickers=tickers_sanitized)
    betas.index = tickers_orig  # Restore original symbols

    sector_dummies = pd.get_dummies(sector_series, prefix="sector")
    return pd.concat([betas, sector_dummies], axis=1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_spy_factors.py <path_to_prices_csv>")
        sys.exit(1)

    filepath = sys.argv[1]
    prices_df = load_prices(filepath)
    panel = build_factor_panel(prices_df)

    print("Done. Panel shape:", panel.shape)
    print(panel.head(10))
