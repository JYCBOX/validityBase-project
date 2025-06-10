# spy500.py
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------

def fetch_constituents() -> list[str]:
    """Return a list of S&P-500 tickers scraped from Wikipedia."""
    wiki = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return wiki["Symbol"].tolist()


def compute_returns_from_csv(path: str) -> pd.DataFrame:
    """
    Load a local CSV of daily returns.
    The CSV must have: a 'date' column      → index
                      and a 'SPY' return column.
    """
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if "SPY" not in df.columns:
        raise KeyError(f"CSV at {path!r} must include a 'SPY' column")
    return df


def compute_returns_from_yfinance(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    """
    Download adjusted-close prices with yfinance and convert to daily returns.

    NOTE: We import yfinance *here* so that the rest of the module can be
    imported even when yfinance is not installed (e.g., in CI test runs).
    """
    try:
        import yfinance as yf                                   # ← lazy-import
    except ImportError as exc:
        raise ImportError(
            "`yfinance` is required only for `compute_returns_from_yfinance` "
            "(pip install yfinance)."
        ) from exc

    yf_tickers = [t.replace(".", "-") for t in tickers] + ["SPY"]
    raw = yf.download(
        yf_tickers,
        start=start,
        end=end,
        progress=False,
        group_by="ticker",
        auto_adjust=True,
    )

    prices = pd.DataFrame(
        {orig: raw[clean]["Close"] for orig, clean in zip(tickers + ["SPY"], yf_tickers)},
        index=raw.index,
    )

    rets = prices.pct_change().dropna(how="all")
    if "SPY" not in rets.columns:
        raise KeyError("Downloaded data missing SPY returns")
    return rets


def compute_betas(rets: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    βᵢ = Cov(rᵢ, r_SPY) / Var(r_SPY)   for each ticker i.
    """
    var_spy = rets["SPY"].var()
    covs = rets.cov().loc[tickers, "SPY"]
    betas = (covs / var_spy).rename("beta")
    betas.index = tickers
    return betas


def build_sector_dummies(tickers: list[str]) -> pd.DataFrame:
    """One-hot encode each ticker’s GICS1 sector, scraped from Wikipedia."""
    wiki = (
        pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
          .set_index("Symbol")
    )
    sectors = wiki.loc[tickers, "GICS Sector"]
    return pd.get_dummies(sectors, prefix="sector")


def build_spy500_panel(rets: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Combine betas and sector dummies into a single DataFrame:

        index   → ticker
        columns → ['beta', 'sector_<name>', …]
    """
    betas = compute_betas(rets, tickers)
    sectors = build_sector_dummies(tickers)
    return pd.concat([betas, sectors], axis=1)


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build SPY-500 factor panel")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--csv", help="Path to a returns CSV (must include SPY column)")
    grp.add_argument(
        "--yf", nargs=2, metavar=("START", "END"),
        help="Fetch returns with yfinance; supply start/end dates (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--out", required=True,
        help="Destination .parquet file for the factor panel",
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
