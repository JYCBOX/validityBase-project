import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def run_cross_sectional_regression(
    asset_returns: pd.Series,
    factor_loadings: pd.DataFrame,
    weights: pd.Series = None,
    robust: bool = True,
    huber_t: float = 1.345,
) -> pd.Series:
    """
    Run a certian period of cross-sectional regression:
        r_i = sum_k x_{i,k} * f_k + u_i
    Optionally downweight outliers using Huber’s T norm.

    Parameters
    ----------
    asset_returns : pd.Series
        Excess returns. Index = asset identifiers.
    factor_loadings : pd.DataFrame
        Exposures for the same period. Index = asset identifiers,
        columns = factor names (e.g. ['F0', 'F1', ...]).
    weights : pd.Series, optional
        Cross-sectional weights (e.g. 1/σ² or market-cap).
        Index = asset identifiers. If None, equal-weight = 1 for all.
    robust : bool, default=True
        If True, use statsmodels.RLM with Huber’s T to downweight outliers.
        If False, use WLS (if weights given) or OLS otherwise.
    huber_t : float, default=1.345
        Tuning constant for Huber’s T.

    Returns
    -------
    pd.Series
        Estimated factor returns ˆf_k for that period.
        Index = factor names (same as factor_loadings.columns).
    """
   
    #common_assets = asset_returns.index.intersection(factor_loadings.index)
    #y = asset_returns.loc[common_assets].astype(float)
    #X = factor_loadings.loc[common_assets].astype(float)

    if not asset_returns.index.equals(factor_loadings.index):

        raise ValueError(
            "Asset indices not matched between factors and returns.\n"
        )
    
    y = asset_returns.astype(float)
    X = factor_loadings.astype(float)

    if weights is None:
        w = pd.Series(1.0, index=factor_loadings.index)
    else:
        w = weights.astype(float)


    if robust:
        huber = sm.robust.norms.HuberT(t=huber_t)
        model = sm.RLM(endog=y.values, exog=X.values, M=huber, weights=w.values)
        results = model.fit()
        params = results.params
    else:
        if weights is not None:
            model = sm.WLS(endog=y.values, exog=X.values, weights=w.values)
            results = model.fit()
            params = results.params
        else:
            model = sm.OLS(endog=y.values, exog=X.values)
            results = model.fit()
            params = results.params

    
    return pd.Series(params, index=X.columns, name="factor_returns")



def run_monthly_factor_returns(
    returns_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    weights_df: pd.DataFrame = None,
    robust: bool = True,
    huber_t: float = 1.345,
) -> pd.DataFrame:
    """
    Loop over each month (column) in returns_df, run cross-sectional regression,
    and collect the estimated factor returns in a DataFrame.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset excess returns. Shape = (n_assets, n_months).
        Index = asset identifiers; columns = period identifiers (e.g. '2023-01').
    exposures_df : pd.DataFrame
        MultiIndex DataFrame of exposures.
        Index = (month, asset); columns = factor names.
    weights_df : pd.DataFrame, optional
        MultiIndex DataFrame of cross-sectional weights (same index structure).
        If None, equal-weight is used.
    robust : bool, default=True
        If True, uses RLM (Huber’s T). If False, WLS/OLS.
    huber_t : float, default=1.345
        Tuning constant for Huber’s T.

    Returns
    -------
    pd.DataFrame
        Estimated factor returns for each period. 
        Shape = (n_periods, n_factors). Index = periods; columns = factor names.
    """
    periods = returns_df.columns.tolist()
    factor_names = exposures_df.columns.tolist()
    results = pd.DataFrame(index=periods, columns=factor_names, dtype=float)

    for period in periods:
        
        r_slice = returns_df[period].dropna()

        
        try:
            e_slice = exposures_df.loc[period]
        except KeyError:
            
            e_slice = exposures_df.xs(key=period, level=0)
        
        e_slice = e_slice.reindex(index=r_slice.index).dropna(how="any")

   
        if weights_df is not None:
            try:
                w_slice = weights_df.loc[period].iloc[:, 0]
            except KeyError:
                w_slice = weights_df.xs(key=period, level=0).iloc[:, 0]
            w_slice = w_slice.reindex(index=r_slice.index).dropna()
        else:
            w_slice = None

      
        if w_slice is not None:
            common = r_slice.index.intersection(e_slice.index).intersection(w_slice.index)
            r_cs = r_slice.loc[common]
            e_cs = e_slice.loc[common]
            w_cs = w_slice.loc[common]
        else:
            common = r_slice.index.intersection(e_slice.index)
            r_cs = r_slice.loc[common]
            e_cs = e_slice.loc[common]
            w_cs = None


        facret = run_cross_sectional_regression(
            asset_returns=r_cs,
            factor_loadings=e_cs,
            weights=w_cs,
            robust=robust,
            huber_t=huber_t,
        )

   
        results.loc[period, facret.index] = facret.values

    return results
