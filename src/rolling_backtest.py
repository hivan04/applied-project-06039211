import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.trading_signal import generate_kalman_signals
from src.backtest import compute_strategy_returns


def walk_forward_backtest(
    dynamic_details,
    weights,
    entry_z=2.0,
    exit_z=0.0,
    obs_cov=1.0,
    is_window=756,
    oos_step=63,
):
    """
    Walk-forward backtest on pre-computed Kalman estimates.

    At each fold the IS window provides rolling z-score context and the OOS
    step is where returns are harvested. Kalman estimates are causal so
    slicing the pre-computed dict introduces no lookahead bias.

    Parameters
    ----------
    dynamic_details : dict
        {(pair_name, obs_cov): DataFrame with alpha_t, beta_t columns}.
    weights : dict
        {pair_name: portfolio weight}.
    entry_z : float
        Z-score entry threshold.
    exit_z : float
        Z-score exit threshold.
    obs_cov : float
        Kalman observation covariance to use.
    is_window : int
        Number of trading days in each IS window. Default 756 (~3 years).
    oos_step : int
        Number of trading days per OOS fold. Default 63 (~1 quarter).

    Returns
    -------
    pd.Series
        Concatenated daily portfolio returns across all OOS folds.
    """
    ref_key = next(
        k for k in dynamic_details
        if np.isclose(float(k[1]), obs_cov) and k[0] in weights
    )
    all_dates = dynamic_details[ref_key].index.sort_values()
    n = len(all_dates)

    fold_returns = []
    start_idx = is_window

    while start_idx + oos_step <= n:
        oos_end_idx  = start_idx + oos_step
        context_dates = all_dates[:oos_end_idx]
        oos_dates     = all_dates[start_idx:oos_end_idx]

        details_fold = {
            k: v.loc[v.index.isin(context_dates)]
            for k, v in dynamic_details.items()
            if np.isclose(float(k[1]), obs_cov) and k[0] in weights
        }

        signals = generate_kalman_signals(details_fold, entry_z=entry_z, exit_z=exit_z)

        pair_rets = {}
        for (pair_name, R), df in signals.items():
            if pair_name not in weights:
                continue
            df = compute_strategy_returns(df.copy())
            oos_slice = df.loc[df.index.isin(oos_dates), "strategy_ret"]
            if len(oos_slice) > 0:
                pair_rets[pair_name] = oos_slice * weights[pair_name]

        if pair_rets:
            fold_returns.append(pd.DataFrame(pair_rets).sum(axis=1))

        start_idx += oos_step

    if not fold_returns:
        return pd.Series(dtype=float, name="strategy_ret")

    return pd.concat(fold_returns).sort_index().rename("strategy_ret")

def walk_forward_refined_backtest(
    dynamic_details,
    weights,
    obs_cov=1.0,
    is_window=262,
    oos_step=63,
    **refined_kwargs,
):
    """
    Walk-forward backtest using the refined signal (generate_refined_kalman_signals_2).
    Returns concatenated daily portfolio returns across all OOS folds.
    """
    from src.refined_trading_signal import generate_refined_kalman_signals_2

    ref_key = next(
        k for k in dynamic_details
        if np.isclose(float(k[1]), obs_cov) and k[0] in weights
    )
    all_dates = dynamic_details[ref_key].index.sort_values()
    n = len(all_dates)

    fold_returns = []
    start_idx = is_window

    while start_idx + oos_step <= n:
        oos_end_idx   = start_idx + oos_step
        context_dates = all_dates[:oos_end_idx]
        oos_dates     = all_dates[start_idx:oos_end_idx]

        details_fold = {
            k: v.loc[v.index.isin(context_dates)]
            for k, v in dynamic_details.items()
            if np.isclose(float(k[1]), obs_cov) and k[0] in weights
        }

        signals_df  = generate_refined_kalman_signals_2(details_fold, **refined_kwargs)
        oos_signals = signals_df[signals_df.index.isin(oos_dates)]

        pair_rets = {}
        for pair_name, group in oos_signals.groupby("pair"):
            if pair_name not in weights:
                continue
            group = group.copy()
            group["spread_change"] = group["spread_t"].diff()
            group["strategy_ret"]  = group["active_position"] * group["spread_change"]
            ret = group["strategy_ret"].dropna()
            if len(ret) > 0:
                pair_rets[pair_name] = ret * weights[pair_name]

        if pair_rets:
            fold_returns.append(pd.DataFrame(pair_rets).sum(axis=1))

        start_idx += oos_step

    if not fold_returns:
        return pd.Series(dtype=float, name="strategy_ret")

    return pd.concat(fold_returns).sort_index().rename("strategy_ret")


def rolling_sharpe(portfolio_ret, window=63, annualisation_factor=252):
    """Rolling annualised Sharpe ratio."""
    roll_mean = portfolio_ret.rolling(window).mean()
    roll_std  = portfolio_ret.rolling(window).std()
    sharpe    = roll_mean / roll_std * np.sqrt(annualisation_factor)
    return sharpe.rename("rolling_sharpe")


def plot_rolling_backtest(portfolio_ret, window=63, figsize=(14, 8)):
    """Cumulative PnL and rolling Sharpe for walk-forward results."""
    cum_pnl = 100 * (1 + portfolio_ret.fillna(0)).cumprod()
    roll_sr  = rolling_sharpe(portfolio_ret, window)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(cum_pnl.index, cum_pnl, color="steelblue", lw=1.5)
    axes[0].axhline(100, color="grey", linestyle=":", lw=0.8)
    axes[0].set_title("Walk-Forward Portfolio Cumulative PnL (Base = 100)")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(roll_sr.index, roll_sr, color="darkorange", lw=1.2)
    axes[1].axhline(0,  color="black", lw=0.8)
    axes[1].axhline(1,  color="grey",  linestyle="--", lw=0.8, label="SR = 1")
    axes[1].fill_between(roll_sr.index, 0, roll_sr,
                         where=roll_sr >= 0, alpha=0.15, color="steelblue")
    axes[1].fill_between(roll_sr.index, 0, roll_sr,
                         where=roll_sr < 0,  alpha=0.15, color="red")
    axes[1].set_title(f"Rolling {window}-Day Annualised Sharpe")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.show()
