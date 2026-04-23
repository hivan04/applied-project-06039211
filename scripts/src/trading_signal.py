import numpy as np
import pandas as pd 
from pykalman import KalmanFilter
import statsmodels.api as sm

def generate_kalman_signals(dynamic_details, entry_z=2.0, exit_z=0.5):
    """
    For each (pair, R) in dynamic_details:
    - compute spread_t
    - standardise into z-score
    - generate trading signals

    Returns:
    -------
    signals_dict : dict
        {(pair_name, R): DataFrame with spread, zscore, position}
    """

    signals_dict = {}

    for key, data in dynamic_details.items():
        df = data.copy()

        # 1. Compute spread (already done, but safe to recompute)
        df["spread_t"] = df["y"] - df["alpha_t"] - df["beta_t"] * df["x"]

        # 2. Standardise (z-score)
        df["zscore"] = (
            df["spread_t"] - df["spread_t"].mean()
        ) / df["spread_t"].std()

        # 3. Generate signals
        df["position"] = 0

        # Entry signals
        df.loc[df["zscore"] > entry_z, "position"] = -1   # short spread
        df.loc[df["zscore"] < -entry_z, "position"] = 1   # long spread

        # Exit signals
        df.loc[df["zscore"].abs() < exit_z, "position"] = 0

        # Forward fill positions
        df["position"] = df["position"].ffill()

        # Store result
        signals_dict[key] = df

    return signals_dict
