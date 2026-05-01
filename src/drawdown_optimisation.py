import numpy as np
import pandas as pd 
import math 

"""
Contents:
1) extractor - extracts the spread_t, position & strategy return from the results dictionary into one dataframe
2) pair_volatility - calculates volatlity after we extracted the required inputs from extractor func
3) calculate_vol - does the actual volatility calculation (uses exponential weighted moving avg for est)
4) drawdown_thresholds - calculates the drawdown threshold for reducing or closing a position
"""

def extractor(
    results_dict,
    spread_col="spread_t",
    position_col="position",
    strategy_ret_col="strategy_ret"
):
    """
    Extract spread_t and position from nested results dictionary,
    calculate no-look-ahead strategy returns, and return one combined DataFrame.

    Expected structure:
        results_dict[strategy_name][(pair_name, obs_cov)] = daily DataFrame
    """

    rows = []

    for strategy_name, signals_dict in results_dict.items():

        for key, df in signals_dict.items():

            pair_name, obs_cov = key

            temp = df.copy()

            if spread_col not in temp.columns:
                raise KeyError(
                    f"'{spread_col}' not found for strategy={strategy_name}, pair={pair_name}, obs_cov={obs_cov}."
                )

            if position_col not in temp.columns:
                raise KeyError(
                    f"'{position_col}' not found for strategy={strategy_name}, pair={pair_name}, obs_cov={obs_cov}."
                )

            # Reset index and standardise the date column name
            temp = temp.reset_index()

            if "Date" in temp.columns:
                temp = temp.rename(columns={"Date": "date"})
            elif "index" in temp.columns:
                temp = temp.rename(columns={"index": "date"})
            elif "date" not in temp.columns:
                raise KeyError(
                    f"Could not find a date column after reset_index() for strategy={strategy_name}, pair={pair_name}, obs_cov={obs_cov}."
                )

            # Keep only required columns
            temp = temp[["date", spread_col, position_col]].copy()

            temp["strategy"] = strategy_name
            temp["pair"] = pair_name
            temp["obs_cov"] = obs_cov

            # Daily spread movement
            temp["spread_change"] = temp[spread_col].diff()

            # No-look-ahead strategy return
            temp[strategy_ret_col] = (
                temp[position_col].shift(1) * temp["spread_change"]
            )

            rows.append(temp)

    strategy_returns_df = pd.concat(rows, ignore_index=True)

    strategy_returns_df = strategy_returns_df[
        [
            "date",
            "strategy",
            "pair",
            "obs_cov",
            spread_col,
            position_col,
            "spread_change",
            strategy_ret_col,
        ]
    ]

    strategy_returns_df = strategy_returns_df.dropna(subset=["strategy_ret"]).copy()

    return strategy_returns_df


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

    # Calculate separately by strategy, pair and obs_cov if obs_cov exists.
    group_cols = []

    if "strategy" in df.columns:
     group_cols.append("strategy")

    group_cols.append(pair_col)

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
        vol = vol.shift(1)

        return vol

    df["pair_strategy_vol"] = (
        df.groupby(group_cols, group_keys=False)
          .apply(calculate_vol)
    )

    df = df.dropna(subset=["pair_strategy_vol"]).copy()

    return df

def drawdown_thresholds(
    strategy_returns_df,
    reduce_k=5.0, # adverse move to reduce exposure
    close_k=10.0, # severe adverse move, so close the trade
    holding_period=10, # We will keep the holding period constant (10 days) for simplicity 
    vol_col="pair_strategy_vol",
    reduce_col="reduce_drawdown_threshold",
    close_col="close_drawdown_threshold"
):
    """
    Create volatility-scaled drawdown thresholds for reducing and closing positions.

    - We keep the reduced & closed volatility multiplier and holding periods constant  
    and use common values for simplicity

    Formula:
        reduce threshold = reduce_k * volatility * sqrt(holding_period)
        close threshold  = close_k  * volatility * sqrt(holding_period)

    Expected columns:
        - date
        - strategy
        - pair
        - obs_cov
        - pair_strategy_vol
    """

    df = strategy_returns_df.copy()

    required_cols = ["date", "strategy", "pair", "obs_cov", vol_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    if close_k <= reduce_k:
        raise ValueError("close_k should usually be greater than reduce_k.")

    df[reduce_col] = (
        reduce_k * df[vol_col] * np.sqrt(holding_period)
    )

    df[close_col] = (
        close_k * df[vol_col] * np.sqrt(holding_period)
    )

    threshold_df = df[
        [
            "date",
            "strategy",
            "pair",
            "obs_cov",
            vol_col,
            reduce_col,
            close_col,
        ]
    ].copy()

    return threshold_df
