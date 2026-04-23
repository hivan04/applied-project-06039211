import numpy as np
import pandas as pd
from pykalman import KalmanFilter

MAX_R = 10

def fit_kalman_filter(y, x, delta=0.001, obs_cov=1.0):
    """
    Fit a vanilla Kalman filter to estimate a time-varying intercept
    and hedge ratio in the model:

        y_t = alpha_t + beta_t * x_t + epsilon_t

    Parameters
    ----------
    y : pd.Series
        Dependent variable price series.
    x : pd.Series
        Independent variable price series.
    delta : float, default 1e-4
        Controls the transition covariance. Smaller values make alpha_t
        and beta_t smoother over time.
    obs_cov : float or list-like, default 1.0
        Observation covariance (measurement noise). Can be:
        - a single value, e.g. 1.0
        - multiple values, e.g. [0.5, 1.0, 5.0]

    Returns
    -------
    pd.DataFrame or dict
        If obs_cov is a single value:
            returns one DataFrame indexed like the aligned input series, containing:
            - y
            - x
            - alpha_t
            - beta_t
            - spread_t

        If obs_cov is list-like:
            returns a dictionary where:
            - key = obs_cov value
            - value = DataFrame for that Kalman filter run
    """
    data = pd.concat([y, x], axis=1).dropna().copy()
    data.columns = ["y", "x"]

    y_vals = data["y"].values
    x_vals = data["x"].values

    obs_mat = np.vstack([np.ones(len(x_vals)), x_vals]).T[:, np.newaxis]
    trans_cov = delta / (1 - delta) * np.eye(2)

    # Check whether obs_cov is scalar or iterable
    if np.isscalar(obs_cov):
        obs_cov_values = [obs_cov]
        single_run = True
    else:
        obs_cov_values = list(obs_cov)
        single_run = False

    results = {}

    for r in obs_cov_values:
        if r <= 0:
            raise ValueError("All observation covariance values must be > 0.")
        if r > MAX_R:
            raise ValueError(f"Observation covariance (R={r}) is too large. Must be <= {MAX_R}.")
        
    """
    Key Assumptions for Kalman Filter:
    - We use a small value for delta because we want the hedge ratio to evolve slowly overtime
    - We start with 1 for our observation covariance coefficient as we want the filter to react 
      to incoming data (this is what makes our trading adapt to future prices and be dynamic)
    - Our model assumes linearity between both assets 
    - θt​=θt−1​+ηt, here we assume a random walk 
    - Observation and state noise relies on a Gaussian distribution
    - Our filter assumes smooth movement towards equilibrium (i.e. no structural breaks)
    """
        
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=r,      # R coefficient
        transition_covariance=trans_cov # Q coefficient
    )

    state_means, state_covs = kf.filter(y_vals)

    result_df = data.copy()
    result_df["alpha_t"] = state_means[:, 0]
    result_df["beta_t"] = state_means[:, 1]
    result_df["spread_t"] = (
        result_df["y"] - result_df["alpha_t"] - result_df["beta_t"] * result_df["x"]
    )
    result_df["obs_cov"] = r

    results[r] = result_df

    if single_run:
        return results[obs_cov_values[0]]

    return results

