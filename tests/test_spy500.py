import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from vbase_utils.factors.spy500 import (
    fetch_constituents,
    compute_returns_from_csv,
    compute_betas,
    build_sector_dummies,
    build_spy500_panel,
)


def test_fetch_constituents():
    ticks = fetch_constituents()
    # basic sanity checks
    assert isinstance(ticks, list)
    assert "AAPL" in ticks      
    assert len(ticks) >= 400    


def test_compute_returns_from_csv(tmp_path):
   
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=3, freq="D"),
        "SPY": [0.01, 0.02, -0.005],
        "AAA": [0.02, 0.04, -0.01]
    })
    csv_file = tmp_path / "rets.csv"
    df.to_csv(csv_file, index=False)

   
    rets = compute_returns_from_csv(str(csv_file))
    assert isinstance(rets, pd.DataFrame)
   
    assert len(rets) == 3
    assert pd.api.types.is_datetime64_any_dtype(rets.index)
   
    assert set(rets.columns) == {"SPY", "AAA"}


def test_compute_returns_from_csv_missing_spy(tmp_path):
    
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02"],
        "AAA": [0.01, 0.02]
    })
    f = tmp_path / "bad.csv"
    df.to_csv(f, index=False)
    with pytest.raises(KeyError):
        compute_returns_from_csv(str(f))


def test_compute_betas_simple():
    
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    spy = pd.Series([1, 2, 3, 4, 5], index=dates)
    aaa = spy * 2
    rets = pd.DataFrame({"SPY": spy, "AAA": aaa})
    res = compute_betas(rets, ["AAA"])
    
    assert pytest.approx(res["AAA"], rel=1e-6) == 2.0


def test_build_sector_dummies(monkeypatch):
    
    dummy = pd.DataFrame({
        "Symbol": ["AAA", "BBB", "CCC"],
        "GICS Sector": ["Tech", "Health Care", "Tech"]
    })
    monkeypatch.setattr(pd, "read_html", lambda url: [dummy])

    sec = build_sector_dummies(["AAA", "BBB", "CCC"])
    
    assert set(sec.columns) == {"sector_Tech", "sector_Health Care"}
   
    assert list(sec.loc["AAA"]) == [1, 0]
    assert list(sec.loc["BBB"]) == [0, 1]
    assert list(sec.loc["CCC"]) == [1, 0]


def test_build_spy500_panel_integration(tmp_path, monkeypatch):
    
    monkeypatch.setattr("vbase_utils.factors.spy500.fetch_constituents",
                        lambda: ["AAA", "BBB"])

    
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    data = {
        "SPY": [1, 2, 3, 4],
        "AAA": [2, 4, 6, 8],
        "BBB": [1, 1, 1, 1],
    }
    rets = pd.DataFrame(data, index=dates)
    monkeypatch.setattr("vbase_utils.factors.spy500.compute_returns_from_csv",
                        lambda path: rets)

    
    sec_df = pd.DataFrame({
        "sector_X": [1, 0],
        "sector_Y": [0, 1],
    }, index=["AAA", "BBB"])
    monkeypatch.setattr("vbase_utils.factors.spy500.build_sector_dummies",
                        lambda ticks: sec_df)

   
    panel = build_spy500_panel(rets, ["AAA", "BBB"])
    
    assert list(panel.columns) == ["beta", "sector_X", "sector_Y"]
    
    assert panel.shape == (2, 3)

    
    assert pytest.approx(panel.loc["AAA", "beta"], rel=1e-6) == 2.0
    
    assert pytest.approx(panel.loc["BBB", "beta"], abs=1e-6) == 0.0
