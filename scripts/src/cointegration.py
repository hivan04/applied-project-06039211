import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint 

"""
HELPERS:
- adf_test
- determine_integration_order
- run_ols
- estimate_ecm
"""

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

    # Integration order checks
    y_order_res = determine_integration_order(y, max_diff=max_diff, regression=regression)
    x_order_res = determine_integration_order(x, max_diff=max_diff, regression=regression)

    y_order = y_order_res["order"]
    x_order = x_order_res["order"]

    results["asset_y_integration"] = y_order_res
    results["asset_x_integration"] = x_order_res
    results["same_order_integration"] = (y_order == x_order)

    # Standard Engle-Granger setup: both series should be I(1)
    if y_order is None or x_order is None:
        results["decision"] = "Integration order could not be determined"
        return results

    if y_order != 1 or x_order != 1:
        results["decision"] = "Both series are not I(1), so standard Engle-Granger is not appropriate"
        return results

    # OLS + cointegration tests
    ols_model, residuals = run_ols(y, x)
    coint_stat, coint_pvalue, _ = coint(y, x)

    residual_adf = adf_test(residuals, regression=regression)
    cointegrated = (
        residual_adf["p_value"] < 0.05 and
        coint_pvalue < 0.05
    )

    params = ols_model.params
    pvalues = ols_model.pvalues

    results.update({
        "ols_summary": {
            "params": params.to_dict(),
            "rsquared": ols_model.rsquared,
            "pvalues": pvalues.to_dict()
        },
        "intercept": params.get("const", np.nan),
        "hedge_ratio": params.get("x", params.iloc[-1]),
        "residual_adf": residual_adf,
        "coint_test_pvalue": coint_pvalue,
        "cointegrated": cointegrated,
        "spread": residuals,
        "y_aligned": y,
        "x_aligned": x
    })

    if not cointegrated:
        results["decision"] = "Residual is not stationary, so the pair is not cointegrated"
        return results

    # ECM
    ecm_model, ecm_df = estimate_ecm(y, x, residuals.shift(1))
    results["ecm_summary"] = {
        "params": ecm_model.params.to_dict(),
        "pvalues": ecm_model.pvalues.to_dict(),
        "rsquared": ecm_model.rsquared
    }
    results["ecm_data"] = ecm_df
    results["decision"] = "Cointegrated"

    return results

# 滚动协整检验（252天窗口）
# Rolling cointegration analysis
def analyze_pair(y_series, x_series, pair_name=None, regression="c", max_diff=2):
    """
    Full workflow for one pair:
    1. Align data
    2. Check both series are I(1)
    3. Run OLS
    4. Test residual stationarity
    5. Cross-check with coint()
    6. Estimate ECM if cointegrated
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

    # Integration order checks
    y_order_res = determine_integration_order(y, max_diff=max_diff, regression=regression)
    x_order_res = determine_integration_order(x, max_diff=max_diff, regression=regression)

    y_order = y_order_res["order"]
    x_order = x_order_res["order"]

    results["asset_y_integration"] = y_order_res
    results["asset_x_integration"] = x_order_res
    results["same_order_integration"] = (y_order == x_order)

    # Standard Engle-Granger setup: both series should be I(1)
    if y_order is None or x_order is None:
        results["decision"] = "Integration order could not be determined"
        return results

    if y_order != 1 or x_order != 1:
        results["decision"] = "Both series are not I(1), so standard Engle-Granger is not appropriate"
        return results

    # Run OLS in levels
    ols_model, residuals = run_ols(y, x)

    hedge_ratio = ols_model.params.get("x", np.nan)
    intercept = ols_model.params.get("const", np.nan)

    results["ols_summary"] = {
        "params": ols_model.params.to_dict(),
        "pvalues": ols_model.pvalues.to_dict(),
        "rsquared": ols_model.rsquared
    }

    # Store OLS outputs
    results["hedge_ratio"] = hedge_ratio
    results["intercept"] = intercept
    results["spread"] = residuals
    results["y_aligned"] = y
    results["x_aligned"] = x

    # Residual stationarity test (Engle-Granger step 2)
    resid_adf = adf_test(residuals, regression=regression)
    results["residual_adf"] = resid_adf

    # Built-in cointegration test cross-check
    coint_stat, coint_pvalue, coint_crit = coint(y, x)
    results["coint_test"] = {
        "test_stat": coint_stat,
        "p_value": coint_pvalue,
        "critical_values": {
            "1%": coint_crit[0],
            "5%": coint_crit[1],
            "10%": coint_crit[2]
        }
    }
    results["coint_test_pvalue"] = coint_pvalue

    # Require both tests to pass
    cointegrated = (
        resid_adf["is_stationary_5pct"] and
        coint_pvalue < 0.05
    )

    if not cointegrated:
        results["decision"] = "Residual is not stationary or coint() does not confirm cointegration"
        return results

    # ECM
    ecm_model, ecm_df = estimate_ecm(y, x, residuals.shift(1))
    results["ecm_summary"] = {
        "params": ecm_model.params.to_dict(),
        "pvalues": ecm_model.pvalues.to_dict(),
        "rsquared": ecm_model.rsquared
    }
    results["ecm_data"] = ecm_df

    # Cointegration accepted
    results["cointegrated"] = True
    results["decision"] = "Cointegration"

    return results

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
        "coint_test_pvalue": res.get("coint_test_pvalue"),
        "ecm_coef": res.get("ecm_summary", {}).get("params", {}).get("ect_lag"),
        "decision": res.get("decision")
    })

    return pd.DataFrame(rows)

