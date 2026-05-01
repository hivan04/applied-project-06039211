import numpy as np
import pandas as pd 
from pykalman import KalmanFilter
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

"""
Contents:
1) generate_refined_kalman_signals - generates refined signals (including dynamic z-score thresholds 
   and drawdowns that have been calculated to each pair based off the pairs volatility)
2) plot_refined_trade_signals - plots the refined trading singals 
3) plot_refined_trade_signals_risk - plots only the risk trading signals (reduced exposure 
   and closed by drawdown)
4) plots_loop - for loop functin to save all the combination of pairs with different R values 
   (R = how much the filter trusts the new observed data point relative to the model’s prior estimate)
5) refined_performance_metrics - calculates the performance metrics for our strategies
6) iter_result_frames - format function that helps to store data values consistently throughout dataframe
"""

def generate_refined_kalman_signals(
    dynamic_details,
    z_window=60,
    band_window=126,
    upper_quantile=0.90,
    lower_quantile=0.10,
    exit_z=0.5,
    drawdown_thresholds=None,
    reduced_exposure=0.5,
    shift_bands=True
):
    """
    Generate refined Kalman trading signals using dynamic rolling z-score bands
    and a volatility-scaled drawdown-based risk management overlay.

    Parameters
    ----------
    dynamic_details : dict
        Dictionary where keys are (pair_name, obs_cov) and values are DataFrames.

    drawdown_thresholds : pd.DataFrame, optional
        DataFrame containing:
        - date
        - pair
        - obs_cov
        - reduce_drawdown_threshold
        - close_drawdown_threshold

    Returns
    -------
    signals_df : pd.DataFrame
        Combined DataFrame containing refined signals for all pairs and R values.
    """

    all_results = []

    # Prepare threshold DataFrame once
    if drawdown_thresholds is not None:
        threshold_df = drawdown_thresholds.copy()
        threshold_df["date"] = pd.to_datetime(threshold_df["date"])

        threshold_df = threshold_df[
            [
                "date",
                "pair",
                "obs_cov",
                "reduce_drawdown_threshold",
                "close_drawdown_threshold"
            ]
        ].copy()
    else:
        threshold_df = None

    for key, data in dynamic_details.items():
        pair_name, obs_cov = key

        df = data.copy()

        # 1. Compute Kalman spread
        df["spread_t"] = df["y"] - df["alpha_t"] - df["beta_t"] * df["x"]

        # 2. Compute rolling z-score
        df["rolling_mean"] = df["spread_t"].rolling(z_window).mean()
        df["rolling_std"] = df["spread_t"].rolling(z_window).std()

        df["zscore"] = (df["spread_t"] - df["rolling_mean"]) / df["rolling_std"]

        # 3. Compute dynamic quantile bands
        df["upper_band"] = df["zscore"].rolling(band_window).quantile(upper_quantile)
        df["lower_band"] = df["zscore"].rolling(band_window).quantile(lower_quantile)

        if shift_bands:
            df["upper_band"] = df["upper_band"].shift(1)
            df["lower_band"] = df["lower_band"].shift(1)

        df = df.dropna().copy()

        # Add identifiers before merging thresholds
        df["pair"] = pair_name
        df["obs_cov"] = obs_cov

        # Create date column for merging
        df = df.reset_index()

        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})

        df["date"] = pd.to_datetime(df["date"])

        # Merge dynamic drawdown thresholds
        if threshold_df is not None:
            df = df.merge(
                threshold_df,
                on=["date", "pair", "obs_cov"],
                how="left"
            )
        else:
            df["reduce_drawdown_threshold"] = np.nan
            df["close_drawdown_threshold"] = np.nan

        # 4. Generate raw trading signal from dynamic bands
        df["raw_position"] = np.nan

        df.loc[df["zscore"] > df["upper_band"], "raw_position"] = -1
        df.loc[df["zscore"] < df["lower_band"], "raw_position"] = 1
        df.loc[df["zscore"].abs() < exit_z, "raw_position"] = 0

        df["raw_position"] = df["raw_position"].ffill().fillna(0)

        # 5. Drawdown-based position overlay
        df["spread_change"] = df["spread_t"].diff()

        position = []
        trade_pnl = []
        trade_equity = []
        running_peak = []
        drawdown = []
        risk_state = []

        current_position = 0.0
        current_trade_equity = 0.0
        current_peak = 0.0

        previous_spread = None

        for _, row in df.iterrows():

            raw_pos = row["raw_position"]
            spread_now = row["spread_t"]

            reduce_dd_t = row["reduce_drawdown_threshold"]
            close_dd_t = row["close_drawdown_threshold"]

            if previous_spread is None:
                pnl = 0.0
                previous_spread = spread_now
            else:
                spread_diff = spread_now - previous_spread

                # Long spread profits when spread rises.
                # Short spread profits when spread falls.
                pnl = current_position * spread_diff

                previous_spread = spread_now

            # Reset trade-level equity when flat
            if raw_pos == 0:
                current_trade_equity = 0.0
                current_peak = 0.0
                current_position = 0.0
                dd = 0.0
                state = "flat"

            # Reset trade-level equity when direction changes
            elif np.sign(raw_pos) != np.sign(current_position) and current_position != 0:
                current_trade_equity = 0.0
                current_peak = 0.0
                current_position = raw_pos
                dd = 0.0
                state = "new_trade"

            else:
                current_trade_equity += pnl
                current_peak = max(current_peak, current_trade_equity)

                dd = current_trade_equity - current_peak

                # Thresholds are positive values, while drawdown is negative.
                if pd.notna(close_dd_t) and dd <= -close_dd_t:
                    current_position = 0.0
                    state = "closed_by_drawdown"

                elif pd.notna(reduce_dd_t) and dd <= -reduce_dd_t:
                    current_position = reduced_exposure * raw_pos
                    state = "reduced_by_drawdown"

                else:
                    current_position = raw_pos
                    state = "normal"

            position.append(current_position)
            trade_pnl.append(pnl)
            trade_equity.append(current_trade_equity)
            running_peak.append(current_peak)
            drawdown.append(dd)
            risk_state.append(state)

        df["position"] = position
        df["trade_pnl"] = trade_pnl
        df["trade_equity"] = trade_equity
        df["running_peak"] = running_peak
        df["trade_drawdown"] = drawdown
        df["risk_state"] = risk_state

        # 6. Add model parameters
        df["z_window"] = z_window
        df["band_window"] = band_window
        df["upper_quantile"] = upper_quantile
        df["lower_quantile"] = lower_quantile
        df["exit_z"] = exit_z
        df["reduced_exposure"] = reduced_exposure

        # Set date back as index if you prefer
        df = df.set_index("date")

        all_results.append(df)

    signals_df = pd.concat(all_results, axis=0)

    return signals_df


def plot_refined_trade_signals(signals_df, pair_name, obs_cov=None, figsize=(14, 9)):
    """
    Visualise refined long-entry, short-entry, and exit signals for one pair.

    Parameters
    ----------
    signals_df : DataFrame
        Output of generate_refined_kalman_signals.
    pair_name : str
        Pair label to plot.
    obs_cov : float, optional
        Observation covariance R. If omitted, the first matching value is used.
    """
    pair_df = signals_df.loc[signals_df["pair"] == pair_name].copy()

    if pair_df.empty:
        raise KeyError(f"Pair '{pair_name}' was not found in the refined signals DataFrame.")

    available_r = sorted(pair_df["obs_cov"].unique().tolist())

    if obs_cov is None:
        selected_obs_cov = available_r[0]
    else:
        selected_obs_cov = None
        for candidate in available_r:
            if np.isclose(float(candidate), float(obs_cov)):
                selected_obs_cov = candidate
                break

        if selected_obs_cov is None:
            raise KeyError(f"obs_cov={obs_cov} was not found. Available values: {available_r}")

    df = pair_df.loc[pair_df["obs_cov"] == selected_obs_cov].copy().sort_index()

    previous_position = df["position"].shift(1).fillna(0)
    long_entries = (df["position"] > 0) & (previous_position <= 0)
    short_entries = (df["position"] < 0) & (previous_position >= 0)
    exits = (df["position"] == 0) & (previous_position != 0)
    reduced = df["risk_state"] == "reduced_by_drawdown"
    forced_closes = df["risk_state"] == "closed_by_drawdown"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    axes[0].plot(df.index, df["spread_t"], color="steelblue", lw=1.4, label="Spread")
    axes[0].scatter(df.index[long_entries], df.loc[long_entries, "spread_t"], color="green", marker="^", s=90, label="Buy Pair / Long Spread")
    axes[0].scatter(df.index[short_entries], df.loc[short_entries, "spread_t"], color="red", marker="v", s=90, label="Sell Pair / Short Spread")
    axes[0].scatter(df.index[exits], df.loc[exits, "spread_t"], color="black", marker="x", s=70, label="Exit")

    if reduced.any():
        axes[0].scatter(df.index[reduced], df.loc[reduced, "spread_t"], color="darkorange", marker="o", s=45, label="Reduced Exposure")

    if forced_closes.any():
        axes[0].scatter(df.index[forced_closes], df.loc[forced_closes, "spread_t"], color="purple", marker="D", s=40, label="Closed by Drawdown")

    axes[0].axhline(0, color="grey", linestyle="--", lw=1)
    axes[0].set_title(f"Refined Trade Signals for {pair_name} | R={selected_obs_cov}")
    axes[0].set_ylabel("Spread")
    axes[0].legend(loc="best")

    axes[1].plot(df.index, df["zscore"], color="darkorange", lw=1.2, label="Z-score")
    axes[1].plot(df.index, df["upper_band"], color="firebrick", linestyle="--", lw=1, label="Upper Entry Band")
    axes[1].plot(df.index, df["lower_band"], color="seagreen", linestyle="--", lw=1, label="Lower Entry Band")
    axes[1].scatter(df.index[long_entries], df.loc[long_entries, "zscore"], color="green", marker="^", s=80)
    axes[1].scatter(df.index[short_entries], df.loc[short_entries, "zscore"], color="red", marker="v", s=80)
    axes[1].scatter(df.index[exits], df.loc[exits, "zscore"], color="black", marker="x", s=60)

    if reduced.any():
        axes[1].scatter(df.index[reduced], df.loc[reduced, "zscore"], color="darkorange", marker="o", s=35)

    if forced_closes.any():
        axes[1].scatter(df.index[forced_closes], df.loc[forced_closes, "zscore"], color="purple", marker="D", s=35)

    exit_z = float(df["exit_z"].iloc[0])
    axes[1].axhline(exit_z, color="grey", linestyle=":", lw=1, label=f"Exit +{exit_z}")
    axes[1].axhline(-exit_z, color="grey", linestyle=":", lw=1, label=f"Exit -{exit_z}")
    axes[1].axhline(0, color="black", linestyle="-", lw=0.8)
    axes[1].set_ylabel("Z-score")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best")

    plt.tight_layout()

def plot_refined_trade_signals_risk(signals_df, pair_name, obs_cov=None, figsize=(14, 9)):
    """
    Visualise drawdown-based risk management events for one pair.

    Parameters
    ----------
    signals_df : DataFrame
        Output of generate_refined_kalman_signals.
    pair_name : str
        Pair label to plot.
    obs_cov : float, optional
        Observation covariance R. If omitted, the first matching value is used.
    """
    pair_df = signals_df.loc[signals_df["pair"] == pair_name].copy()

    if pair_df.empty:
        raise KeyError(f"Pair '{pair_name}' was not found in the refined signals DataFrame.")

    available_r = sorted(pair_df["obs_cov"].unique().tolist())

    if obs_cov is None:
        selected_obs_cov = available_r[0]
    else:
        selected_obs_cov = None
        for candidate in available_r:
            if np.isclose(float(candidate), float(obs_cov)):
                selected_obs_cov = candidate
                break

        if selected_obs_cov is None:
            raise KeyError(f"obs_cov={obs_cov} was not found. Available values: {available_r}")

    df = pair_df.loc[pair_df["obs_cov"] == selected_obs_cov].copy().sort_index()

    reduced = df["risk_state"] == "reduced_by_drawdown"
    forced_closes = df["risk_state"] == "closed_by_drawdown"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # -------------------
    # Top panel: spread
    # -------------------
    axes[0].plot(df.index, df["spread_t"], color="steelblue", lw=1.4, label="Spread")

    if reduced.any():
        axes[0].scatter(
            df.index[reduced],
            df.loc[reduced, "spread_t"],
            color="darkorange",
            marker="o",
            s=45,
            label="Reduced Exposure"
        )

    if forced_closes.any():
        axes[0].scatter(
            df.index[forced_closes],
            df.loc[forced_closes, "spread_t"],
            color="purple",
            marker="D",
            s=40,
            label="Closed by Drawdown"
        )

    axes[0].axhline(0, color="grey", linestyle="--", lw=1)
    axes[0].set_title(f"Risk Management Overlay for {pair_name} | R={selected_obs_cov}")
    axes[0].set_ylabel("Spread")
    axes[0].legend(loc="best")

    # -------------------
    # Bottom panel: z-score
    # -------------------
    axes[1].plot(df.index, df["zscore"], color="darkorange", lw=1.2, label="Z-score")
    axes[1].plot(df.index, df["upper_band"], color="firebrick", linestyle="--", lw=1, label="Upper Entry Band")
    axes[1].plot(df.index, df["lower_band"], color="seagreen", linestyle="--", lw=1, label="Lower Entry Band")

    if reduced.any():
        axes[1].scatter(
            df.index[reduced],
            df.loc[reduced, "zscore"],
            color="darkorange",
            marker="o",
            s=35,
            label="Reduced Exposure"
        )

    if forced_closes.any():
        axes[1].scatter(
            df.index[forced_closes],
            df.loc[forced_closes, "zscore"],
            color="purple",
            marker="D",
            s=35,
            label="Closed by Drawdown"
        )

    exit_z = float(df["exit_z"].iloc[0])
    axes[1].axhline(exit_z, color="grey", linestyle=":", lw=1, label=f"Exit +{exit_z}")
    axes[1].axhline(-exit_z, color="grey", linestyle=":", lw=1, label=f"Exit -{exit_z}")
    axes[1].axhline(0, color="black", linestyle="-", lw=0.8)

    axes[1].set_ylabel("Z-score")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best")

    plt.tight_layout()

def plots_loop(
    signals_df,
    save_dir,
    plot_func,
    dpi=300
):
    """
    Save refined trade signal plots for each pair and obs_cov combination.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Combined refined signal DataFrame containing 'pair' and 'obs_cov'.

    save_dir : str
        Directory where plots will be saved.

    plot_func : function
        Plotting function, e.g. plot_refined_trade_signals.

    dpi : int
        Resolution for saved figures.
    """

    os.makedirs(save_dir, exist_ok=True)

    plt.ioff()

    pair_r_combinations = (
        signals_df[["pair", "obs_cov"]]
        .drop_duplicates()
        .sort_values(["pair", "obs_cov"])
        .itertuples(index=False, name=None)
    )

    for pair_name, obs_cov in pair_r_combinations:

        plot_func(
            signals_df=signals_df,
            pair_name=pair_name,
            obs_cov=obs_cov
        )

        clean_pair_name = (
            str(pair_name)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        clean_obs_cov = str(obs_cov).replace(".", "p")

        filename = f"refined_{clean_pair_name}_R{clean_obs_cov}.png"
        filepath = os.path.join(save_dir, filename)

        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close()

    print(f"Saved all refined trade signal plots to: {save_dir}")


def refined_performance_metrics(
    results_dict,
    rf=None,
    initial_capital=1.0,
    annualisation_factor=252
):
    performance_rows = []

    def iter_result_frames(results_obj):
        """
        Normalise supported refined-result containers into a common iterator of
        (strategy_key, pair_name, obs_cov, df).

        Supported inputs
        ----------------
        1. DataFrame returned by generate_refined_kalman_signals
        2. Flat dict: {(pair_name, obs_cov): df}
        3. Nested dict: {strategy_key: {(pair_name, obs_cov): df}}
        """
        if isinstance(results_obj, pd.DataFrame):
            if not {"pair", "obs_cov"}.issubset(results_obj.columns):
                raise ValueError(
                    "Refined results DataFrame must contain 'pair' and 'obs_cov' columns."
                )

            strategy_key = (
                results_obj["strategy"].iloc[0]
                if "strategy" in results_obj.columns
                else "refined_dynamic_bands"
            )

            for (pair_name, obs_cov), df in results_obj.groupby(["pair", "obs_cov"]):
                yield strategy_key, pair_name, obs_cov, df.copy()
            return

        if not isinstance(results_obj, dict):
            raise TypeError(
                "results_dict must be a DataFrame, a flat dict, or a nested dict of DataFrames."
            )

        if not results_obj:
            return

        first_key = next(iter(results_obj))
        first_value = results_obj[first_key]

        if (
            isinstance(first_key, tuple)
            and len(first_key) == 2
            and isinstance(first_value, pd.DataFrame)
        ):
            for (pair_name, obs_cov), df in results_obj.items():
                yield "refined_dynamic_bands", pair_name, obs_cov, df.copy()
            return

        for strategy_key, signals_dict in results_obj.items():
            if isinstance(signals_dict, pd.DataFrame):
                if not {"pair", "obs_cov"}.issubset(signals_dict.columns):
                    raise ValueError(
                        "Nested refined results DataFrame must contain 'pair' and 'obs_cov' columns."
                    )

                for (pair_name, obs_cov), df in signals_dict.groupby(["pair", "obs_cov"]):
                    yield strategy_key, pair_name, obs_cov, df.copy()
                continue

            if not isinstance(signals_dict, dict):
                raise TypeError(
                    "Nested refined results must map strategy names to dicts or DataFrames."
                )

            for (pair_name, obs_cov), df in signals_dict.items():
                yield strategy_key, pair_name, obs_cov, df.copy()

    # Prepare risk-free rate if provided
    if rf is not None:
        rf = rf.copy()

        # If date/Date is a column, set it as the index
        if "Date" in rf.columns:
            rf["Date"] = pd.to_datetime(rf["Date"])
            rf = rf.set_index("Date")
        elif "date" in rf.columns:
            rf["date"] = pd.to_datetime(rf["date"])
            rf = rf.set_index("date")

        rf = rf.sort_index()

        # Use the first column as the RF series
        if isinstance(rf, pd.DataFrame):
            rf_series = rf.iloc[:, 0]
        else:
            rf_series = rf

        rf_series = rf_series.astype(float)

    for strategy_key, pair_name, R, df in iter_result_frames(results_dict):

        # Ensure Date is available and correctly formatted
        if "Date" not in df.columns:
            df = df.reset_index()

        if "date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"date": "Date"})

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df.set_index("Date", drop=False)

        # Calculate refined strategy returns if needed
        if "strategy_return" not in df.columns:

            df["spread_change"] = df["spread_t"].diff()

            # Lag position to avoid look-ahead bias
            df["lagged_position"] = df["position"].shift(1)

            df["strategy_pnl"] = (
                df["lagged_position"] * df["spread_change"]
            )

            df["strategy_pnl"] = df["strategy_pnl"].fillna(0)

            df["strategy_return"] = (
                df["strategy_pnl"] / initial_capital
            )

        else:
            # Keep this available for later metrics
            if "strategy_pnl" not in df.columns:
                df["strategy_pnl"] = df["strategy_return"] * initial_capital

            if "lagged_position" not in df.columns:
                df["lagged_position"] = df["position"].shift(1)

        strategy_returns = df["strategy_return"].dropna()

        if len(strategy_returns) == 0:
            continue

        # Risk-free adjustment
        if rf is not None:
            rf_aligned = rf_series.reindex(strategy_returns.index).ffill()

            # Drop observations where RF is still missing
            valid_idx = rf_aligned.dropna().index

            strategy_returns = strategy_returns.loc[valid_idx]
            rf_aligned = rf_aligned.loc[valid_idx]

            excess_returns = strategy_returns - rf_aligned
        else:
            excess_returns = strategy_returns

        if len(excess_returns) == 0:
            continue

        # Core return metrics
        avg_return = strategy_returns.mean() * 100
        avg_excess_return = excess_returns.mean() * 100

        total_pnl = df.loc[strategy_returns.index, "strategy_pnl"].sum()
        total_return = total_pnl / initial_capital

        annualised_return = excess_returns.mean() * annualisation_factor
        annualised_volatility = excess_returns.std() * np.sqrt(annualisation_factor)

        # Sharpe calculated using decimal excess returns
        std_excess_return = excess_returns.std()

        sharpe = (
            excess_returns.mean() / std_excess_return
            if std_excess_return != 0
            else np.nan
        )

        annualised_sharpe = (
            sharpe * np.sqrt(annualisation_factor)
            if not np.isnan(sharpe)
            else np.nan
        )

        # Additional refined metrics
        hit_rate = (strategy_returns > 0).mean() * 100

        num_trades = df["position"].diff().abs().sum() / 2

        average_pnl = df.loc[strategy_returns.index, "strategy_pnl"].mean()

        if "trade_drawdown" in df.columns:
            max_trade_drawdown = df["trade_drawdown"].min()
        else:
            max_trade_drawdown = np.nan

        n_obs = len(strategy_returns)

        n_active_days = (
            df.loc[strategy_returns.index, "lagged_position"]
            .abs()
            .gt(0)
            .sum()
        )

        active_day_ratio = (
            n_active_days / n_obs * 100
            if n_obs > 0
            else np.nan
        )

        # Extract strategy parameters
        try:
            entry_z = float(strategy_key.split("_")[1])
        except Exception:
            entry_z = np.nan

        try:
            exit_z = float(strategy_key.split("_")[3])
        except Exception:
            exit_z = df["exit_z"].iloc[0] if "exit_z" in df.columns else np.nan

        # Optional parameters from df if available
        z_window = df["z_window"].iloc[0] if "z_window" in df.columns else np.nan
        band_window = df["band_window"].iloc[0] if "band_window" in df.columns else np.nan
        upper_quantile = df["upper_quantile"].iloc[0] if "upper_quantile" in df.columns else np.nan
        lower_quantile = df["lower_quantile"].iloc[0] if "lower_quantile" in df.columns else np.nan

        performance_rows.append({
            "strategy": strategy_key,
            "pair": pair_name,
            "obs_cov": R,

            "entry_z": entry_z,
            "exit_z": exit_z,
            "z_window": z_window,
            "band_window": band_window,
            "upper_quantile": upper_quantile,
            "lower_quantile": lower_quantile,

            "average_pnl": average_pnl,
            "total_pnl": total_pnl,

            "avg_return (%)": avg_return,
            "avg_excess_return (%)": avg_excess_return,
            "total_return (%)": total_return * 100,
            "annualised_return (%)": annualised_return * 100,
            "annualised_volatility (%)": annualised_volatility * 100,

            "sharpe": sharpe,
            "annualised_sharpe": annualised_sharpe,

            "hit_rate (%)": hit_rate,
            "num_trades": num_trades,
            "max_trade_drawdown": max_trade_drawdown,

            "n_obs": n_obs,
            "n_active_days": n_active_days,
            "active_day_ratio (%)": active_day_ratio
        })

    return pd.DataFrame(performance_rows)
