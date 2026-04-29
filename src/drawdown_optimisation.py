import numpy as np
import pandas as pd 
import math 

def pair_volatility(
    signals_df,
    return_col="strategy_ret",
    pair_col="pair",
    obs_cov_col="obs_cov",
    method="ewma",
    window=60,
    annualise=False,
    annualisation_factor=252,
    min_periods=None
):
    """
    Add pair-level strategy return volatility to a signals DataFrame.

    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame containing strategy returns for each pair.
        Expected columns include:
        - pair_col, e.g. "pair"
        - return_col, e.g. "strategy_ret"
        - optionally obs_cov_col, e.g. "obs_cov"

    return_col : str
        Column containing daily strategy returns.

    pair_col : str
        Column identifying the pair.

    obs_cov_col : str
        Column identifying the Kalman observation covariance parameter.
        If present, volatility is calculated separately for each pair and obs_cov.

    method : {"ewma", "rolling", "expanding"}
        Volatility estimation method.

    window : int
        Rolling/EWMA window length.
        For EWMA, this is used as the span.

    annualise : bool
        If True, annualises the volatility by multiplying by sqrt(annualisation_factor).
        For your drawdown rule, this should usually be False.

    annualisation_factor : int
        Number of trading periods per year. Usually 252 for daily data.

    min_periods : int or None
        Minimum observations required.
        If None, defaults to:
        - window for rolling
        - 2 for EWMA/expanding

    Returns
    -------
    pd.DataFrame
        Copy of signals_df with a new column:
        - "pair_strategy_vol"
    """

    df = signals_df.copy()

    if return_col not in df.columns:
        raise KeyError(f"'{return_col}' column not found in signals_df.")

    if pair_col not in df.columns:
        raise KeyError(f"'{pair_col}' column not found in signals_df.")

    if min_periods is None:
        min_periods = window if method == "rolling" else 2

    # Calculate separately by pair and obs_cov if obs_cov exists.
    group_cols = [pair_col]
    if obs_cov_col in df.columns:
        group_cols.append(obs_cov_col)

    def calculate_vol(group):
        returns = group[return_col].astype(float)

        if method == "ewma":
            vol = returns.ewm(
                span=window,
                adjust=False,
                min_periods=min_periods
            ).std()

        elif method == "rolling":
            vol = returns.rolling(
                window=window,
                min_periods=min_periods
            ).std()

        elif method == "expanding":
            vol = returns.expanding(
                min_periods=min_periods
            ).std()

        else:
            raise ValueError("method must be one of: 'ewma', 'rolling', or 'expanding'.")

        if annualise:
            vol = vol * np.sqrt(annualisation_factor)

        return vol

    df["pair_strategy_vol"] = (
        df.groupby(group_cols, group_keys=False)
          .apply(calculate_vol)
    )

    return df
