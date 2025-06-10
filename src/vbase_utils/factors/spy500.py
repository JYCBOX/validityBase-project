
from __future__ import annotations
import pandas as pd
import yfinance as yf
import numpy as np


def fetch_constituents() -> list[str]:
    """Pull the S&P 500 tickers from Wikipedia."""
    wiki = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return wiki["Symbol"].tolist()


def compute_returns_from_csv(path: str) -> pd.DataFrame:
    """
    Load a local CSV of daily returns.
    Must include a 'SPY' column and have 'date' as the index.
    """
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if "SPY" not in df.columns:
        raise KeyError(f"CSV at {path!r} must include a 'SPY' column")
    return df


def compute_returns_from_yfinance(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance and convert to daily returns.
    Automatically fixes tickers like BRK.B → BRK-B.
    """
    yf_tickers = [t.replace(".", "-") for t in tickers] + ["SPY"]
    raw = yf.download(
        yf_tickers,
        start=start,
        end=end,
        progress=False,
        group_by="ticker",
        auto_adjust=True
    )
   
    prices = pd.DataFrame({
        orig: raw[clean]["Close"]
        for orig, clean in zip(tickers + ["SPY"], yf_tickers)
    }, index=raw.index)
    rets = prices.pct_change().dropna(how="all")
    if "SPY" not in rets.columns:
        raise KeyError("Downloaded data missing SPY returns")
    return rets


def compute_betas(
    rets: pd.DataFrame, tickers: list[str]
) -> pd.Series:
    """
    Compute each ticker’s beta versus SPY:
        βᵢ = Cov(rᵢ, r_SPY) / Var(r_SPY)
    """
    var_spy = rets["SPY"].var()
    covs    = rets.cov().loc[tickers, "SPY"]
    betas   = (covs / var_spy).rename("beta")
    betas.index = tickers
    return betas


def build_sector_dummies(tickers: list[str]) -> pd.DataFrame:
    """
    One-hot encode each ticker’s GICS1 sector by scraping Wikipedia.
    """
    wiki = (
        pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
          .set_index("Symbol")
    )
    sectors = wiki.loc[tickers, "GICS Sector"]
    return pd.get_dummies(sectors, prefix="sector")


def build_spy500_panel(
    rets: pd.DataFrame,
    tickers: list[str]
) -> pd.DataFrame:
    """
    Combine betas and sector dummies into one wide DataFrame:
        index = ticker
        columns = ['beta', 'sector_<name>', ...]
    """
    betas   = compute_betas(rets, tickers)
    sectors = build_sector_dummies(tickers)
    return pd.concat([betas, sectors], axis=1)


def main():
    """CLI entry point: build and save the SPY500 factor panel."""
    import argparse

    parser = argparse.ArgumentParser(description="Build SPY500 factor panel")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--csv", help="Path to a returns CSV file (must include SPY column)"
    )
    group.add_argument(
        "--yf", nargs=2, metavar=("START","END"),
        help="Use yfinance to fetch returns; specify start/end dates"
    )
    parser.add_argument(
        "--out", required=True,
        help="Path where the output parquet panel will be written"
    )
    args = parser.parse_args()

    tickers = fetch_constituents()
    if args.csv:
        rets = compute_returns_from_csv(args.csv)
    else:
        rets = compute_returns_from_yfinance(tickers, *args.yf)

    panel = build_spy500_panel(rets, tickers)
    panel.to_parquet(args.out)
    print(f"Wrote panel {panel.shape} to {args.out}")


if __name__ == "__main__":
    main()
