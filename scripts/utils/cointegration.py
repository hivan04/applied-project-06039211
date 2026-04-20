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

    y_order_res = determine_integration_order(y, max_diff=max_diff, regression=regression)
    x_order_res = determine_integration_order(x, max_diff=max_diff, regression=regression)

    y_order = y_order_res["order"]
    x_order = x_order_res["order"]

    results["asset_y_integration"] = y_order_res
    results["asset_x_integration"] = x_order_res
    results["same_order_integration"] = (y_order == x_order)

    if y_order is None or x_order is None:
        results["decision"] = "Integration order could not be determined"
        return results

    if y_order != 1 or x_order != 1:
        results["decision"] = "Both series are not I(1), so standard Engle-Granger is not appropriate"
        return results

    ols_model, residuals = run_ols(y, x)
    residual_adf = adf_test(residuals, regression=regression)
    cointegrated = residual_adf["is_stationary_5pct"]

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
        "cointegrated": cointegrated,
        "spread": residuals,
        "y_aligned": y,
        "x_aligned": x
    })

    if not cointegrated:
        results["decision"] = "Residual is not stationary, so the pair is not cointegrated"
        return results

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
def rolling_cointegration(
    y_series,
    x_series,
    pair_name=None,
    formation_window=252,
    step=21,
    regression="c",
    max_diff=2,
    return_details=False
):
    """
    Run Engle-Granger cointegration tests over rolling formation windows.

    Parameters
    ----------
    y_series, x_series : array-like or pandas Series
        Price series for one pair.
    pair_name : str, optional
        Label used in the output table.
    formation_window : int, default 252
        Number of observations in each rolling formation window.
    step : int, default 21
        Number of observations to move forward after each window.
    regression : str, default "c"
        Regression setting passed to the ADF tests.
    max_diff : int, default 2
        Maximum differencing order used in integration-order checks.
    return_details : bool, default False
        If True, return both the summary table and full per-window results.

    Returns
    -------
    pandas DataFrame
        One row per rolling window. If return_details=True, returns
        (summary_df, detailed_results).
    """
    if formation_window <= 0:
        raise ValueError("formation_window must be positive")

    if step <= 0:
        raise ValueError("step must be positive")

    df = pd.concat(
        [
            pd.Series(y_series).rename("y"),
            pd.Series(x_series).rename("x")
        ],
        axis=1
    ).dropna()

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    summary_rows = []
    detailed_results = {}

    for start in range(0, len(df) - formation_window + 1, step):
        end = start + formation_window
        window_df = df.iloc[start:end]
        window_start = window_df.index[0]
        window_end = window_df.index[-1]
        window_label = f"{window_start}_{window_end}"

        result = analyze_pair(
            y_series=window_df["y"],
            x_series=window_df["x"],
            pair_name=pair_name,
            regression=regression,
            max_diff=max_diff
        )

        result["window_start"] = window_start
        result["window_end"] = window_end
        result["window_start_pos"] = start
        result["window_end_pos"] = end - 1
        detailed_results[window_label] = result

        summary_rows.append({
            "pair": pair_name,
            "window_start": window_start,
            "window_end": window_end,
            "window_start_pos": start,
            "window_end_pos": end - 1,
            "n_obs": result.get("n_obs"),
            "y_order": result.get("asset_y_integration", {}).get("order"),
            "x_order": result.get("asset_x_integration", {}).get("order"),
            "cointegrated": result.get("cointegrated", False),
            "hedge_ratio": result.get("hedge_ratio"),
            "intercept": result.get("intercept"),
            "residual_adf_pvalue": result.get("residual_adf", {}).get("p_value"),
            "ecm_coef": result.get("ecm_summary", {}).get("params", {}).get("ect_lag"),
            "ecm_pvalue": result.get("ecm_summary", {}).get("pvalues", {}).get("ect_lag"),
            "decision": result.get("decision")
        })

    summary_df = pd.DataFrame(summary_rows)

    if return_details:
        return summary_df, detailed_results

    return summary_df

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

