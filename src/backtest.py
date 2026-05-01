import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

"""
Contents:
1) compute_strategy_returns - computes strategy return
2) plot_trade_signals - plots trade signals (visuals of buy/sell/exit)
3) performance_metrics - computes performance metrics of our strategies
"""


# Calculate returns for sensitivity analysis 
def compute_strategy_returns(df):
    df = df.copy()

    # Spread change
    df["spread_ret"] = df["spread_t"].diff()

    # Strategy return (lag position to avoid lookahead bias)
    df["strategy_ret"] = df["position"].shift(1) * df["spread_ret"]

    return df


# Plotting buy and sell positions of trade signals (with z-score thresholds)
def plot_trade_signals(results_dict, strategy_name, pair_name, obs_cov=None, figsize=(14, 9)):
    """
    Visualise long-entry, short-entry, and exit signals for one pair.

    Parameters
    ----------
    results_dict : dict
        Nested dictionary such as results[strategy_name][(pair_name, obs_cov)] = signal_df.
    strategy_name : str
        Strategy key such as 'entry_1.0_exit_0.5'.
    pair_name : str
        Pair label used in the signal dictionary.
    obs_cov : float, optional
        Observation covariance R. If omitted, the first matching pair is used.
    """
    strategy_signals = results_dict[strategy_name]
    matching_keys = [key for key in strategy_signals if key[0] == pair_name]

    if not matching_keys:
        raise KeyError(f"Pair '{pair_name}' was not found under strategy '{strategy_name}'.")

    if obs_cov is None:
        selected_key = matching_keys[0]
    else:
        selected_key = None
        for key in matching_keys:
            if np.isclose(float(key[1]), float(obs_cov)):
                selected_key = key
                break

        if selected_key is None:
            available_r = [key[1] for key in matching_keys]
            raise KeyError(f"obs_cov={obs_cov} was not found. Available values: {available_r}")

    df = strategy_signals[selected_key].copy().sort_index()

    entry_z = float(strategy_name.split("_")[1])
    exit_z = float(strategy_name.split("_")[3])

    previous_position = df["position"].shift(1).fillna(0)
    long_entries = (df["position"] == 1) & (previous_position != 1)
    short_entries = (df["position"] == -1) & (previous_position != -1)
    exits = (df["position"] == 0) & (previous_position != 0)

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
    axes[0].axhline(0, color="grey", linestyle="--", lw=1)
    axes[0].set_title(f"Trade Signals for {pair_name} | R={selected_key[1]} | {strategy_name}")
    axes[0].set_ylabel("Spread")
    axes[0].legend(loc="best")

    axes[1].plot(df.index, df["zscore"], color="darkorange", lw=1.2, label="Z-score")
    axes[1].scatter(df.index[long_entries], df.loc[long_entries, "zscore"], color="green", marker="^", s=80)
    axes[1].scatter(df.index[short_entries], df.loc[short_entries, "zscore"], color="red", marker="v", s=80)
    axes[1].scatter(df.index[exits], df.loc[exits, "zscore"], color="black", marker="x", s=60)
    axes[1].axhline(entry_z, color="red", linestyle="--", lw=1, label=f"Entry +{entry_z}")
    axes[1].axhline(-entry_z, color="green", linestyle="--", lw=1, label=f"Entry -{entry_z}")
    axes[1].axhline(exit_z, color="grey", linestyle=":", lw=1, label=f"Exit +{exit_z}")
    axes[1].axhline(-exit_z, color="grey", linestyle=":", lw=1, label=f"Exit -{exit_z}")
    axes[1].axhline(0, color="black", linestyle="-", lw=0.8)
    axes[1].set_ylabel("Z-score")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best")

    plt.tight_layout()
    # plt.show() - Don't include because there are too many plots 

# Function for creating dataframe of performance metrics

def performance_metrics(results_dict, rf=None, annualisation_factor=252):
    performance_rows = []

    # Prepare risk-free rate if provided
    if rf is not None:
        rf = rf.copy()

        # If date is a column, set it as the index
        if "date" in rf.columns:
            rf["date"] = pd.to_datetime(rf["date"])
            rf = rf.set_index("date")

        rf = rf.sort_index()

        # Use the first column as the RF series
        if isinstance(rf, pd.DataFrame):
            rf_series = rf.iloc[:, 0]
        else:
            rf_series = rf

        rf_series = rf_series.astype(float)

    for strategy_key, signals_dict in results_dict.items():

        for (pair_name, R), df in signals_dict.items():

            df = df.copy()

            if "strategy_ret" not in df.columns:
                df = compute_strategy_returns(df)

            strategy_returns = df["strategy_ret"].dropna()

            if len(strategy_returns) == 0:
                continue

            # Align risk-free rate to strategy returns
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

            # Display returns in percentage terms
            avg_return = strategy_returns.mean() * 100
            avg_excess_return = excess_returns.mean() * 100

            # Total cumulative strategy return/pnl in raw decimal units
            total_pnl = strategy_returns.sum()

            # Sharpe should be calculated using decimal returns, not percentage returns
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

            # Hit rate based on daily positive strategy returns
            hit_rate = (strategy_returns > 0).mean() * 100

            # Approximate number of round-trip trades
            num_trades = df["position"].diff().abs().sum() / 2

            performance_rows.append({
                "strategy": strategy_key,
                "pair": pair_name,
                "obs_cov": R,
                "entry_z": float(strategy_key.split("_")[1]),
                "exit_z": float(strategy_key.split("_")[3]),
                "avg_return (%)": avg_return,
                "avg_excess_return (%)": avg_excess_return,
                "total_pnl": total_pnl,
                "sharpe": sharpe,
                "annualised_sharpe": annualised_sharpe,
                "hit_rate (%)": hit_rate,
                "num_trades": num_trades
            })

    return pd.DataFrame(performance_rows)

