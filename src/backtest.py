import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.bootstrap import bootstrap_sharpe_ci
from src.config import TRANSACTION_COST_PER_LEG, TRANSACTION_LEGS

"""
Contents:
1) compute_strategy_returns - computes strategy return
2) leg_transaction_costs - per-pair turnover-based transaction costs
3) plot_trade_signals - plots trade signals (visuals of buy/sell/exit)
4) performance_metrics - computes performance metrics of our strategies
"""


# Calculate returns for sensitivity analysis
def compute_strategy_returns(df):
    df = df.copy()

    # Spread change
    df["spread_ret"] = df["spread_t"].diff()

    # Strategy return (lag position to avoid lookahead bias)
    df["strategy_ret"] = df["position"].shift(1) * df["spread_ret"]

    return df


# Per-pair transaction costs (in spread-return units), driven by position turnover.
def leg_transaction_costs(position, cost_per_leg=TRANSACTION_COST_PER_LEG, n_legs=TRANSACTION_LEGS):
    """
    Transaction cost per period for a single pair, charged on position turnover.

    A pairs trade has ``n_legs`` legs (default 2: long one name, short the other),
    so each unit of |Δposition| transacts every leg once. Entering or exiting a
    full position costs ``n_legs * cost_per_leg``; a flip (+1 -> -1) costs double;
    partial adjustments (e.g. 1.0 -> 0.5 exposure) are charged pro-rata.

    Parameters
    ----------
    position : pd.Series
        Position/exposure series for one pair (e.g. -1, 0, +1, or fractional).
    cost_per_leg : float
        Fractional cost per leg per unit turnover (0.0015 = 15 bps).
    n_legs : int
        Number of legs transacted per unit of position (2 for a pair).

    Returns
    -------
    pd.Series
        Non-negative cost aligned to ``position`` (same index).
    """
    pos = pd.Series(position).astype(float).fillna(0.0)
    turnover = pos.diff().abs()
    if len(turnover):
        # Opening the very first position also incurs a cost.
        turnover.iloc[0] = abs(pos.iloc[0])
    return turnover * n_legs * cost_per_leg


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

# Portfolio-level performance metrics (takes a plain return Series)
def performance_metrics(
    ret,
    positions=None,
    rf=None,
    cost_per_side=0.002,
    annualisation_factor=252,
    ret_type="returns",
    initial_capital=None,
    bootstrap_ci=False,
    n_bootstrap=1000,
    bootstrap_seed=None,
):
    ret = ret.copy().dropna()
    pre_tc_ret = ret.copy()

    # Transaction costs
    if positions is not None:
        pos = positions.reindex(ret.index).fillna(0)
        position_changes = pos.diff().abs().fillna(pos.abs())
        tc = position_changes * cost_per_side
    else:
        # Charge on days the portfolio transitions between active and flat
        transitions = (ret != 0).astype(int).diff().abs().fillna(0).astype(bool)
        tc = transitions.astype(float) * cost_per_side

    ret = ret - tc

    # Risk-free rate alignment
    if rf is not None:
        if isinstance(rf, pd.DataFrame):
            rf = rf.iloc[:, 0]
        rf = rf.astype(float).reindex(ret.index).ffill()
        if rf.mean() > 0.1:   # likely annualised (e.g. 0.05 = 5%)
            rf = rf / annualisation_factor
        excess_ret = ret - rf
    else:
        excess_ret = ret

    if ret_type == "pnl":
        # Raw P&L path: do NOT use (1+ret).cumprod()
        total_pnl    = ret.sum()
        cum_pnl      = ret.cumsum()
        rolling_max  = cum_pnl.cummax()
        drawdown     = cum_pnl - rolling_max
        max_drawdown = drawdown.min()

        if initial_capital is not None:
            avg_ret_out    = ret.mean() / initial_capital * 100
            avg_excess_out = excess_ret.mean() / initial_capital * 100
            total_pnl_out  = total_pnl / initial_capital * 100
            max_dd_out     = max_drawdown / initial_capital * 100
        else:
            avg_ret_out    = ret.mean()
            avg_excess_out = excess_ret.mean()
            total_pnl_out  = total_pnl
            max_dd_out     = max_drawdown

    else:
        # Proportional return path
        cum_ret      = (1 + ret).cumprod()
        rolling_max  = cum_ret.cummax()
        drawdown     = (cum_ret - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        total_pnl    = 100 * (cum_ret.iloc[-1] - 1)

        avg_ret_out    = ret.mean() * 100
        avg_excess_out = excess_ret.mean() * 100
        total_pnl_out  = total_pnl
        max_dd_out     = max_drawdown

    # Sharpe
    sharpe     = excess_ret.mean() / excess_ret.std(ddof=1) if excess_ret.std(ddof=1) != 0 else np.nan
    ann_sharpe = sharpe * np.sqrt(annualisation_factor) if not np.isnan(sharpe) else np.nan

    if bootstrap_ci and not np.isnan(ann_sharpe):
        ci = bootstrap_sharpe_ci(
            excess_ret,
            n_bootstrap=n_bootstrap,
            annualisation_factor=annualisation_factor,
            seed=bootstrap_seed,
        )
        sharpe_display = f"{ann_sharpe:.4f}  [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
    else:
        sharpe_display = round(ann_sharpe, 4)

    # Hit rate — always on pre-TC signal to measure signal quality
    active   = pre_tc_ret[pre_tc_ret != 0]
    hit_rate = (active > 0).mean() * 100 if len(active) > 0 else np.nan

    return pd.Series({
        "avg_return (%)":        round(avg_ret_out, 4),
        "avg_excess_return (%)": round(avg_excess_out, 4),
        "annualised_sharpe":     sharpe_display,
        "total_pnl":             round(total_pnl_out, 2),
        "hit_rate (%)":          round(hit_rate, 2),
        "max_drawdown (%)":      round(max_dd_out, 2),
        "num_obs":               len(ret),
    })

# Combining plot
def plot_combined_pnl(
    is_results,
    oos_results,
    strategy_name="entry_2.0_exit_0.0",
    obs_cov=1.0,
    figsize=(14, 3),
    save_path=None,
    dpi=300
):
    """
    Plot IS and OOS cumulative PnL side by side per pair, both indexed to 100.
 
    Each row is one pair. Left column = In-Sample, Right column = Out-of-Sample.
    Both periods start at base 100 so y-axes are directly comparable within
    each period without large OOS moves squashing the IS curves.
 
    Nothing in is_results or oos_results is modified — all scaling is local
    to this function.
 
    Parameters
    ----------
    is_results : dict
        Nested dict of shape {strategy_name: {(pair_name, obs_cov): df}}.
        Same object used throughout notebook 5_backtest.
    oos_results : dict
        Same structure as is_results for the OOS period.
    strategy_name : str
        Key into is_results / oos_results, e.g. "entry_2.0_exit_0.0".
    obs_cov : float
        Observation covariance R value to plot.
    figsize : tuple
        (width, height_per_pair). Total figure height scales with pair count.
    save_path : str or None
        If provided, saves the figure to this filepath instead of showing it.
        Include the filename, e.g. "outputs/combined_pnl.png".
    dpi : int
        Resolution for saved figure.
 
    Usage
    -----
    # Interactive
    plot_combined_pnl(is_results, oos_results)
 
    # Save to file
    plot_combined_pnl(
        is_results, oos_results,
        strategy_name="entry_2.0_exit_0.0",
        obs_cov=1.0,
        save_path=PROJECT_ROOT / "outputs/combined_pnl_entry2_exit0_R1.png"
    )
    """
 
    is_signals  = is_results[strategy_name]
    oos_signals = oos_results[strategy_name]
 
    all_pairs = sorted(set(k[0] for k in is_signals))
    n = len(all_pairs)
 
    # 2 columns: IS on left, OOS on right
    fig, axes = plt.subplots(
        n, 2,
        figsize=(figsize[0], figsize[1] * n),
        sharex=False
    )
 
    # Ensure axes is always 2D even for a single pair
    if n == 1:
        axes = [axes]
 
    for row_idx, pair_name in enumerate(all_pairs):
 
        ax_is  = axes[row_idx][0]
        ax_oos = axes[row_idx][1]
 
        # Match pair + obs_cov in both dicts
        is_key  = next(
            (k for k in is_signals  if k[0] == pair_name and np.isclose(float(k[1]), obs_cov)),
            None
        )
        oos_key = next(
            (k for k in oos_signals if k[0] == pair_name and np.isclose(float(k[1]), obs_cov)),
            None
        )
 
        if is_key is None or oos_key is None:
            ax_is.set_title(f"{pair_name}  [R={obs_cov} not found]")
            continue
 
        is_df  = compute_strategy_returns(is_signals[is_key].copy())
        oos_df = compute_strategy_returns(oos_signals[oos_key].copy())
 
        # Both periods independently indexed to base 100
        is_cumret  = 100 * (1 + is_df["strategy_ret"].fillna(0)).cumprod()
        oos_cumret = 100 * (1 + oos_df["strategy_ret"].fillna(0)).cumprod()
 
        # --- IS plot (left) ---
        ax_is.plot(is_cumret.index, is_cumret, color="steelblue", lw=1.5)
        ax_is.axhline(100, color="grey", linestyle=":", lw=0.8)
        ax_is.set_title(f"{pair_name}\nIn-Sample", fontsize=9)
        ax_is.set_ylabel("Portfolio Value (Base = 100)")
        ax_is.grid(True, alpha=0.3)
        ax_is.set_xlabel("Date")
 
        # --- OOS plot (right) ---
        ax_oos.plot(oos_cumret.index, oos_cumret, color="darkorange", lw=1.5)
        ax_oos.axhline(100, color="grey", linestyle=":", lw=0.8)
        ax_oos.set_title(f"{pair_name}\nOut-of-Sample", fontsize=9)
        ax_oos.set_ylabel("Portfolio Value (Base = 100)")
        ax_oos.grid(True, alpha=0.3)
        ax_oos.set_xlabel("Date")
 
    fig.suptitle(
        f"Cumulative PnL  |  {strategy_name}  |  R = {obs_cov}",
        fontsize=12,
        y=1.01
    )
    plt.tight_layout()
 
    if save_path is not None:
        os.makedirs(
            os.path.dirname(str(save_path)) if os.path.dirname(str(save_path)) else ".",
            exist_ok=True
        )
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved to: {save_path}")
    else:
        plt.show()
