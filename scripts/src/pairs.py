import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import statsmodels.api as sm

from src.kalman import *
# from kalman import *

MAX_R = 10

"""
Start by creating the simple hedge ratio first and save the results into .xslx or something
Then form the dynamic model and import the kalman filter from the kalman.py script
"""

# Static hedge ratio (using OLS)
def estimate_hedge_ratio_df(y, x, pair_name=None, add_intercept=True):
    """
    Estimate hedge ratio using OLS and return:
    - summary DataFrame (1 row)
    - spread Series
    """

    data = pd.concat([y, x], axis=1).dropna()
    data.columns = ["y", "x"]

    y_aligned = data["y"]
    x_aligned = data["x"]

    if add_intercept:
        X = sm.add_constant(x_aligned)
        model = sm.OLS(y_aligned, X).fit()
        alpha = model.params["const"]
        beta = model.params["x"]
        spread = y_aligned - (alpha + beta * x_aligned)
    else:
        model = sm.OLS(y_aligned, x_aligned).fit()
        alpha = 0.0
        beta = model.params["x"]
        spread = y_aligned - beta * x_aligned

    # Summary as DataFrame (1 row)
    summary_df = pd.DataFrame([{
        "pair": pair_name,
        "alpha": alpha,
        "beta": beta,
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs)
    }])

    return summary_df, spread, model

# Dynamic hedge ratio (using Kalman Filter)
def estimate_dynamic_hedge_ratio_df(y, x, pair_name=None, delta=1e-4, obs_cov=1.0):
    """
    Estimate dynamic hedge ratio using Kalman filter and return:
    - summary DataFrame
    - spread Series
    - full DataFrame
    """
    data = fit_kalman_filter(y=y, x=x, delta=delta, obs_cov=obs_cov).copy()

    data["spread_t"] = data["y"] - data["alpha_t"] - data["beta_t"] * data["x"]

    summary_df = pd.DataFrame([{
        "pair": pair_name,
        "obs_cov (R)": obs_cov,
        "alpha_t_last": data["alpha_t"].iloc[-1],
        "beta_t_last": data["beta_t"].iloc[-1],
        "beta_t_mean": data["beta_t"].mean(),
        "beta_t_std": data["beta_t"].std()
    }])

    spread = data["spread_t"]

    return summary_df, spread, data



