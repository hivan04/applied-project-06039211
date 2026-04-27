import numpy as np
import pandas as pd 
from pykalman import KalmanFilter
import statsmodels.api as sm

def generate_refined_kalman_signals(dynamic_details, entry_z=2.0, exit_z=0.5):
    """
    Generate trading signals using z-score thresholds with proper position holding.

    Parameters
    ----------
    dynamic_details : dict
        {(pair_name, R): DataFrame with columns y, x, alpha_t, beta_t}
    entry_z : float
        Entry threshold for z-score
    exit_z : float
        Exit threshold for z-score

    Returns
    -------
    signals_dict : dict
        {(pair_name, R): DataFrame with spread, zscore, position}
    """

    signals_dict = {}

    for key, data in dynamic_details.items():
        df = data.copy()

        # 1. Compute spread
        df["spread_t"] = df["y"] - df["alpha_t"] - df["beta_t"] * df["x"]

        # 2. Compute z-score
        df["rolling_mean"] = df["spread_t"].rolling(60).mean()
        df["rolling_std"] = df["spread_t"].rolling(60).std()
        
        df["zscore"] = (df["spread_t"] - df["rolling_mean"]) / df["rolling_std"]
        df = df.dropna()

        # 3. Initialise with NaN (IMPORTANT)
        df["position"] = np.nan

        # 4. Entry signals
        df.loc[df["zscore"] > entry_z, "position"] = -1   # short spread
        df.loc[df["zscore"] < -entry_z, "position"] = 1   # long spread

        # 5. Exit signals
        df.loc[df["zscore"].abs() < exit_z, "position"] = 0

        # 6. Forward-fill positions and replace initial NaNs with 0
        df["position"] = df["position"].ffill().fillna(0)

        # Store result
        signals_dict[key] = df

    return signals_dict


