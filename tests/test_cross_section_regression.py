import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cross_section_regression import (
    run_cross_sectional_regression,
    run_monthly_factor_returns,
)

def make_simple_cs_data():
    # f = 0.5
    asset_returns = pd.Series([0.5, 1.0], index=["A", "B"])
    factor_loadings = pd.DataFrame([[1.0], [2.0]],
                                   index=["A", "B"],
                                   columns=["F"])
    true_f = (asset_returns.values @ np.linalg.pinv(factor_loadings.values)).item()
    return asset_returns, factor_loadings, true_f

def make_monthly_data(n_assets=5, n_factors=2, n_months=3, seed=0):
    np.random.seed(seed)
    assets = [f"AS{i}" for i in range(n_assets)]
    factors = [f"F{j}" for j in range(n_factors)]
    periods = [f"2025-0{m+1}" for m in range(n_months)]
    idx = pd.MultiIndex.from_product([periods, assets],
                                     names=["period","asset"])
    # exposures
    exposures = pd.DataFrame(np.random.randn(len(idx), n_factors),
                             index=idx, columns=factors)
    true_fr = pd.DataFrame(np.random.randn(n_months, n_factors)*0.1,
                           index=periods, columns=factors)
    # returns_df
    ret = {}
    for per in periods:
        X = exposures.loc[per].values
        f = true_fr.loc[per].values
        noise = np.random.randn(n_assets)*1e-3
        ret[per] = pd.Series(X.dot(f) + noise, index=assets)
    returns = pd.DataFrame(ret)
    return returns, exposures, true_fr

# ——— Tests for run_cross_sectional_regression —————————————————

def test_ols_no_weights():
    y, X, true_f = make_simple_cs_data()
    est = run_cross_sectional_regression(y, X,
                                         weights=None,
                                         robust=False)
    assert pytest.approx(true_f, rel=1e-6) == est["F"]

def test_wls_with_weights_does_not_error():
    y, X, _ = make_simple_cs_data()
    weights = pd.Series([1.0, 0.5], index=["A", "B"])
    est = run_cross_sectional_regression(y, X,
                                         weights=weights,
                                         robust=False)
    assert isinstance(est["F"], float)

def test_index_mismatch_raises_value_error():
    y = pd.Series([0.1, 0.2], index=["A", "B"])
    X = pd.DataFrame([[1],[2]], index=["A", "C"], columns=["F"])
    with pytest.raises(ValueError):
        run_cross_sectional_regression(y, X)

# ——— Tests for run_monthly_factor_returns —————————————————

def test_monthly_returns_shape_and_accuracy():
    returns, exposures, true_fr = make_monthly_data()
    est = run_monthly_factor_returns(returns,
                                     exposures,
                                     weights_df=None,
                                     robust=True)
    assert list(est.columns) == list(true_fr.columns)
    assert list(est.index)   == list(true_fr.index)
    max_diff = (est - true_fr).abs().values.max()
    assert max_diff < 1e-2

def test_monthly_returns_missing_period_yields_nan():
    returns, exposures, _ = make_monthly_data()
    exposures2 = exposures.drop("2025-02", level=0)
    est2 = run_monthly_factor_returns(returns,
                                      exposures2,
                                      weights_df=None)
    assert est2.loc["2025-02"].isna().all()