import numpy as np
import pandas as pd 
from pykalman import KalmanFilter
import statsmodels.api as sm
import matplotlib.pyplot as plt

def generate_refined_kalman_signals(
    dynamic_details,
    z_window=60,
    band_window=126,
    upper_quantile=0.90,
    lower_quantile=0.10,
    exit_z=0.5,
    reduce_dd=-0.10,
    close_dd=-1.00,
    reduced_exposure=0.5,
    shift_bands=True
):
    """
    Generate refined Kalman trading signals using dynamic rolling z-score bands
    and a drawdown-based risk management overlay.

    Returns
    -------
    signals_df : DataFrame
        Combined DataFrame containing refined signals for all pairs and R values.
    """

    all_results = []

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

        # 4. Generate raw trading signal from dynamic bands
        df["raw_position"] = np.nan

        df.loc[df["zscore"] > df["upper_band"], "raw_position"] = -1
        df.loc[df["zscore"] < df["lower_band"], "raw_position"] = 1
        df.loc[df["zscore"].abs() < exit_z, "raw_position"] = 0

        df["raw_position"] = df["raw_position"].ffill().fillna(0)

        # 5. Drawdown-based position overlay
        df["spread_change"] = df["spread_t"].diff()

        # Create lists for each of the indicators to assist with determining what the signal should do
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

                if dd <= close_dd:
                    current_position = 0.0
                    state = "closed_by_drawdown"

                elif dd <= reduce_dd:
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

        # 6. Add identifiers so everything is stored in one DataFrame
        df["pair"] = pair_name
        df["obs_cov"] = obs_cov
        df["z_window"] = z_window
        df["band_window"] = band_window
        df["upper_quantile"] = upper_quantile
        df["lower_quantile"] = lower_quantile
        df["exit_z"] = exit_z
        df["reduce_dd"] = reduce_dd
        df["close_dd"] = close_dd
        df["reduced_exposure"] = reduced_exposure

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
