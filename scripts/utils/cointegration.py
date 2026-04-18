import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# ADF test
def adf_test(series, maxlag=None, regression="c", autolag="AIC"):
    """
    Run the Augmented Dickey-Fuller test on a series.
    """
    series = pd.Series(series).dropna()

    if len(series) < 10:
        return {
            "test_stat": np.nan,
            "p_value": np.nan,
            "used_lag": np.nan,
            "n_obs": len(series),
            "critical_values": {},
            "is_stationary_5pct": False
        }

    result = adfuller(series, maxlag=maxlag, regression=regression, autolag=autolag)

    return {
        "test_stat": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary_5pct": result[1] < 0.05
    }

# Determine integration order
def determine_integration_order(series, max_diff=2, regression="c"):
    """
    Determine the order of integration I(d) using repeated ADF tests.
    """
    tests = {}
    current_series = pd.Series(series).dropna()

    for d in range(max_diff + 1):
        test_result = adf_test(current_series, regression=regression)
        tests[f"I({d})"] = test_result

        if test_result["is_stationary_5pct"]:
            return {
                "order": d,
                "tests": tests
            }

        current_series = current_series.diff().dropna()

    return {
        "order": None,
        "tests": tests
    }


# OLS for Engle-Granger step 1
def run_ols(y, x):
    """
    Run OLS: y_t = alpha + beta*x_t + u_t
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    residuals = model.resid

    return model, residuals

# Error Correction Model
def estimate_ecm(y, x, ect_lag):
    """
    Estimate a simple ECM:
        Δy_t = α + βΔx_t + γECT_{t-1} + ε_t
    """
    ecm_df = pd.concat(
        [
            y.diff().rename("dy"),
            x.diff().rename("dx"),
            ect_lag.rename("ect_lag")
        ],
        axis=1
    ).dropna()

    X = sm.add_constant(ecm_df[["dx", "ect_lag"]])
    model = sm.OLS(ecm_df["dy"], X).fit()

    return model, ecm_df

# Analyze one pair
def analyze_pair(y_series, x_series, pair_name=None, regression="c", max_diff=2):
    """
    Full workflow for one pair:
    1. Align data
    2. Check both series are I(1)
    3. Run OLS
    4. Test residual stationarity
    5. Estimate ECM if cointegrated
    """
    df = pd.concat(
        [
            pd.Series(y_series).rename("y"),
            pd.Series(x_series).rename("x")
        ],
        axis=1
    ).dropna()

    y = df["y"]
    x = df["x"]

    results = {
        "pair_name": pair_name,
        "n_obs": len(df),
        "cointegrated": False
    }

    if len(df) < 20:
        results["decision"] = "Insufficient observations after alignment"
        return results

# Analyze all pairs
def analyze_all_pairs(asset1_dict, asset2_dict, regression="c", max_diff=2):
    """
    Run the full workflow for all common keys in the two dictionaries.
    """
    common_keys = sorted(set(asset1_dict).intersection(asset2_dict))
    all_results = {}

    for key in common_keys:
        all_results[key] = analyze_pair(
            y_series=asset1_dict[key],
            x_series=asset2_dict[key],
            pair_name=key,
            regression=regression,
            max_diff=max_diff
        )

    return all_results

# Extract cointegrated pairs for trading
def extract_trading_pairs(results_dict):
    """
    Extract only cointegrated pairs using stored results.
    No re-estimation is performed.
    """
    trading_pairs = {}

    for pair, res in results_dict.items():
        if not res.get("cointegrated", False):
            continue

        trading_pairs[pair] = {
            "hedge_ratio": res.get("hedge_ratio"),
            "intercept": res.get("intercept"),
            "spread": res.get("spread"),
            "y": res.get("y_aligned"),
            "x": res.get("x_aligned"),
            "ecm_coef": res.get("ecm_summary", {}).get("params", {}).get("ect_lag"),
            "ecm_pvalue": res.get("ecm_summary", {}).get("pvalues", {}).get("ect_lag"),
            "residual_adf_pvalue": res.get("residual_adf", {}).get("p_value")
        }

    return trading_pairs

# Summary table
def summarize_results(results_dict):
    rows = []

    for pair, res in results_dict.items():
        rows.append({
            "pair": pair,
            "n_obs": res.get("n_obs"),
            "y_order": res.get("asset_y_integration", {}).get("order"),
            "x_order": res.get("asset_x_integration", {}).get("order"),
            "cointegrated": res.get("cointegrated", False),
            "hedge_ratio": res.get("hedge_ratio"),
            "residual_adf_pvalue": res.get("residual_adf", {}).get("p_value"),
            "ecm_coef": res.get("ecm_summary", {}).get("params", {}).get("ect_lag"),
            "decision": res.get("decision")
        })

    return pd.DataFrame(rows)