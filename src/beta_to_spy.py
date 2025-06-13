"""
Compute 252-day trailing betas of assets to SPY using robust regression.

This script uses pit_robust_betas from vbase_utils.
"""

import pandas as pd
from vbase_utils.stats.pit_robust_betas import pit_robust_betas


def compute_beta_to_spy(
    asset_returns: pd.DataFrame,
    spy_returns: pd.DataFrame,
    min_obs: int = 252,
    half_life: float = 63,
    lambda_: float = 0.97,
) -> pd.DataFrame:
    """
    Computes trailing betas of each asset to SPY using robust regression.

    Args:
        asset_returns (pd.DataFrame): Daily returns of assets, shape (T, N)
        spy_returns (pd.DataFrame): Daily returns of SPY, shape (T, 1)
        min_obs (int): Minimum required lookback window (default 252)
        half_life (float): Half-life for exponential weighting (default 63)
        lambda_ (float): Decay factor for EWMA (default 0.97)

    Returns:
        pd.DataFrame: Beta values indexed by timestamp and asset
    """
    spy_returns = spy_returns.rename(columns={spy_returns.columns[0]: "SPY"})

    results = pit_robust_betas(
        df_asset_rets=asset_returns,
        df_fact_rets=spy_returns,
        half_life=half_life,
        lambda_=lambda_,
        min_timestamps=min_obs,
    )

    df_betas = results["df_betas"]
    df_betas = df_betas.loc[pd.IndexSlice[:, "SPY"], :]  # keep only SPY column
    df_betas.index = df_betas.index.droplevel("factor")  # drop 'SPY' from index
    return df_betas
