import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
from io import StringIO
from typing import Dict, Optional

# STEP 1: Get S&P 500 symbols and GICS sectors
def get_sp500_symbols_and_sectors() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    tables = pd.read_html(StringIO(response.text))
    df = tables[0][["Symbol", "GICS Sector"]].rename(columns={"Symbol": "symbol", "GICS Sector": "sector"})
    return df

# STEP 2: One-hot encode GICS sectors as binary (1/0)
def generate_sector_exposures(sp500_df: pd.DataFrame) -> pd.DataFrame:
    sector_ohe = pd.get_dummies(sp500_df["sector"], prefix="Sector").astype(int)
    return pd.concat([sp500_df[["symbol"]], sector_ohe], axis=1).set_index("symbol")

# STEP 3: Robust regression to estimate betas
def robust_betas(asset_rets, fact_rets, min_timestamps=252):
    results = pd.DataFrame(index=fact_rets.columns, columns=asset_rets.columns, dtype=float)
    for asset in asset_rets:
        y = asset_rets[asset].dropna()
        x = fact_rets.loc[y.index]
        if len(y) >= min_timestamps:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            results.loc[fact_rets.columns[0], asset] = model.params[1]
    return results

# STEP 4: Point-in-time rolling regression
def pit_robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    min_timestamps: int = 252,
) -> Dict[str, pd.DataFrame]:
    if df_asset_rets.empty or df_fact_rets.empty:
        raise ValueError("Input DataFrames cannot be empty.")
    if not isinstance(df_asset_rets.index, pd.DatetimeIndex):
        raise ValueError("df_asset_rets must have a DatetimeIndex.")
    if not df_asset_rets.index.equals(df_fact_rets.index):
        raise ValueError("Indices must align.")
    
    time_index = df_asset_rets.index[min_timestamps:]
    betas = pd.DataFrame(
        index=pd.MultiIndex.from_product([time_index, df_fact_rets.columns], names=["timestamp", "factor"]),
        columns=df_asset_rets.columns,
        dtype=float,
    )
    
    for t in time_index:
        past_assets = df_asset_rets.loc[:t].iloc[-min_timestamps:]
        past_factors = df_fact_rets.loc[:t].iloc[-min_timestamps:]
        beta_matrix = robust_betas(past_assets, past_factors, min_timestamps)
        for factor in df_fact_rets.columns:
            for asset in df_asset_rets.columns:
                betas.loc[(t, factor), asset] = beta_matrix.loc[factor, asset]
    
    return {"df_betas": betas}

# STEP 5: Main process
rets = pd.read_csv("us_stocks_1d_rets.csv", index_col=0, parse_dates=[0])
rets.index = pd.to_datetime(rets.index, utc=True)
rets.index.name = "date"

sp500_df = get_sp500_symbols_and_sectors()
sector_exposures = generate_sector_exposures(sp500_df)

tickers = [t for t in sp500_df["symbol"] if t in rets.columns]
if "SPY" not in rets.columns:
    raise ValueError("SPY not found in daily returns data.")

df_asset_rets = rets[tickers]
df_fact_rets = rets[["SPY"]]

# Run rolling regression
result = pit_robust_betas(df_asset_rets, df_fact_rets, min_timestamps=252)
df_betas = result["df_betas"]

# Save outputs
df_betas.to_csv("betas_vs_spy.csv")
sector_exposures.loc[tickers].to_csv("sector_exposures.csv")

print("Finished: betas_vs_spy.csv and sector_exposures.csv saved.")
