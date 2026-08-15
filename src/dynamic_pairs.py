"""
dynamic_pairs.py — Dynamic (rolling) pair *selection* walk-forward.

The notebook-8 walk-forward rolls only the *trading context*: the pair universe
is fixed once (from ``selected_pairs.pkl`` / ``data/portfolio``) and only the
Kalman hedge ratio adapts. That isolates the effect of the dynamic hedge ratio,
but it cannot answer whether the *pairs themselves* should be re-chosen as
cointegration relationships decay.

This module re-runs Engle-Granger cointegration on a rolling FORMATION window
(default ~12 months), selects every pair that is cointegrated at ``alpha``
(default 0.10, the project threshold), ranks them by cointegration p-value,
greedily enforces unique legs (a stock cannot anchor two pairs in the same
period), then TRADES the chosen pairs over the following TRADING window
(default ~3 months) with the same Kalman + z-score signal used everywhere else.
It then rolls forward and re-selects.

The same engine runs in a STATIC-selection mode (pass ``fixed_pairs``): the pair
set is frozen but the fold structure, dates, costs and z-bands are identical.
Running both modes gives an apples-to-apples Static-vs-Dynamic comparison where
the *only* difference is whether pairs are re-selected — the controlled-variable
comparison the dissertation needs.

No look-ahead:
  * Selection uses ONLY formation-window prices (strictly past data).
  * The Kalman filter is causal and is re-fit per fold on
    formation+trading; z-scores over the trading slice are warmed up by the
    formation context (rolling-60), so no trading-period information leaks back.
  * Transaction costs are charged on the full context position and then sliced
    to the trading dates, so the first trading day is charged relative to the
    prior (formation) position — matching src/rolling_backtest.py.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint

from src.kalman import fit_kalman_filter
from src.trading_signal import generate_kalman_signals
from src.backtest import compute_strategy_returns, leg_transaction_costs
from src.config import (
    TRANSACTION_COST_PER_LEG,
    TRANSACTION_LEGS,
    OBS_COV,
    ENTRY_Z,
    EXIT_Z,
)

# Defaults for the rolling formation / trading schedule (trading days).
FORMATION_WINDOW = 252   # ~12-month formation period for cointegration testing
TRADING_PERIOD   = 63    # ~3-month holding/trading period
SELECTION_ALPHA  = 0.10  # cointegration p-value threshold (project-wide 0.10)


def _pair_name(y_name, x_name):
    """Match the naming convention used across the project's signal dicts."""
    return f"{y_name} vs {x_name}"


def select_pairs_formation(
    formation_prices,
    candidate_pairs,
    alpha=SELECTION_ALPHA,
    top_n=None,
    unique_legs=True,
):
    """
    Rank cointegrated pairs inside one formation window.

    Parameters
    ----------
    formation_prices : pd.DataFrame
        Wide price frame (columns = tickers) restricted to the formation window.
    candidate_pairs : list of (y_name, x_name)
        Ticker pairs to test (already restricted to the desired universe,
        e.g. within-sector combinations).
    alpha : float
        Keep a pair only if the Engle-Granger coint() p-value < alpha.
    top_n : int or None
        Cap on how many pairs to keep after ranking. None keeps all passing.
    unique_legs : bool
        If True, greedily skip a pair whose y or x is already used by a
        higher-ranked (lower p-value) selected pair this period.

    Returns
    -------
    pd.DataFrame
        One row per PASSING pair (sorted by p-value ascending), columns:
        y, x, pair, coint_pvalue, hedge_ratio, selected (bool).
    """
    rows = []
    for y_name, x_name in candidate_pairs:
        pair_df = formation_prices[[y_name, x_name]].dropna()
        # Need enough overlap to run a meaningful cointegration test.
        if len(pair_df) < 30:
            continue
        y = pair_df[y_name]
        x = pair_df[x_name]
        try:
            _, pvalue, _ = coint(y, x)
        except Exception:
            continue
        if not np.isfinite(pvalue) or pvalue >= alpha:
            continue
        # Hedge ratio for the trade sizing / reporting (OLS slope, no intercept
        # term needed here — the Kalman filter re-estimates alpha_t/beta_t).
        beta = np.polyfit(x.values, y.values, 1)[0]
        rows.append({
            "y": y_name,
            "x": x_name,
            "pair": _pair_name(y_name, x_name),
            "coint_pvalue": pvalue,
            "hedge_ratio": beta,
        })

    if not rows:
        return pd.DataFrame(
            columns=["y", "x", "pair", "coint_pvalue", "hedge_ratio", "selected"]
        )

    ranked = pd.DataFrame(rows).sort_values("coint_pvalue").reset_index(drop=True)

    selected = []
    used = set()
    for _, r in ranked.iterrows():
        if unique_legs and (r["y"] in used or r["x"] in used):
            selected.append(False)
            continue
        if top_n is not None and sum(selected) >= top_n:
            selected.append(False)
            continue
        selected.append(True)
        used.update([r["y"], r["x"]])

    ranked["selected"] = selected
    return ranked


def _fold_returns(
    prices,
    selected,
    context_dates,
    trading_dates,
    entry_z,
    exit_z,
    obs_cov,
    cost_per_leg,
    n_legs,
):
    """
    Trade one fold's selected pairs and return equal-weighted daily portfolio
    returns over the trading dates (net of transaction costs).
    """
    pair_rets = {}
    for _, r in selected.iterrows():
        y_name, x_name = r["y"], r["x"]
        pair_name = r["pair"]

        # Kalman is re-fit per fold on the full context (formation + trading);
        # slicing the causal estimates afterwards introduces no look-ahead.
        ctx = prices.loc[prices.index.isin(context_dates), [y_name, x_name]].dropna()
        if len(ctx) < 70:  # need rolling-60 warmup + a few trading days
            continue
        kf_df = fit_kalman_filter(ctx[y_name], ctx[x_name], obs_cov=obs_cov)

        details = {(pair_name, obs_cov): kf_df}
        signals = generate_kalman_signals(details, entry_z=entry_z, exit_z=exit_z)
        sig = compute_strategy_returns(signals[(pair_name, obs_cov)].copy())

        # Cost on the full context so the first trading day is charged relative
        # to the prior (formation) position, then slice to the trading window.
        cost = leg_transaction_costs(sig["position"], cost_per_leg, n_legs)
        net = sig["strategy_ret"].fillna(0.0) - cost
        oos = net.loc[net.index.isin(trading_dates)]
        if len(oos) > 0:
            pair_rets[pair_name] = oos

    if not pair_rets:
        return pd.Series(dtype=float)

    # Equal weight across the pairs held this fold.
    ret_df = pd.DataFrame(pair_rets)
    return ret_df.mean(axis=1)


def dynamic_pair_walk_forward(
    prices_by_sector,
    fixed_pairs=None,
    formation_window=FORMATION_WINDOW,
    trading_period=TRADING_PERIOD,
    alpha=SELECTION_ALPHA,
    top_n=None,
    unique_legs=True,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
    obs_cov=OBS_COV,
    cost_per_leg=TRANSACTION_COST_PER_LEG,
    n_legs=TRANSACTION_LEGS,
):
    """
    Walk-forward backtest with rolling pair RE-SELECTION.

    At each step a ``formation_window`` of prices is used to re-run Engle-Granger
    cointegration across every within-sector candidate pair, select the pairs
    that cointegrate at ``alpha`` (ranked by p-value, unique legs), then trade
    them over the next ``trading_period`` days. The window then advances by
    ``trading_period`` and pairs are re-selected. Non-overlapping trading
    windows tile the out-of-sample timeline exactly once.

    Parameters
    ----------
    prices_by_sector : dict[str, pd.DataFrame]
        {sector_name: wide price frame (columns = tickers)}. Candidate pairs are
        formed WITHIN each sector only. Frames should already be the full
        timeline (IS + OOS concatenated) with a shared DatetimeIndex.
    fixed_pairs : list of (y_name, x_name), optional
        If given, selection is SKIPPED and this fixed set is traded every fold
        (static-selection mode). Use this to produce the controlled Static
        benchmark on the identical fold schedule.
    formation_window, trading_period : int
        Formation and trading lengths in trading days.
    alpha, top_n, unique_legs : see select_pairs_formation.
    entry_z, exit_z, obs_cov, cost_per_leg, n_legs : signal / cost parameters
        (shared with the rest of the pipeline via src.config).

    Returns
    -------
    portfolio_ret : pd.Series
        Concatenated daily equal-weighted portfolio returns across all folds,
        named 'strategy_ret'.
    selection_log : pd.DataFrame
        One row per (fold, selected pair) with formation dates, trading dates,
        p-value and hedge ratio — the audit trail for the dissertation's
        "which pairs, when, and why" discussion.
    """
    # Unified, sorted timeline across all sectors.
    all_index = sorted(set().union(*[df.index for df in prices_by_sector.values()]))
    all_dates = pd.DatetimeIndex(all_index)
    n = len(all_dates)

    # Candidate pairs are within-sector combinations, tagged with their sector.
    candidate_pairs_by_sector = {
        sector: list(combinations(df.columns, 2))
        for sector, df in prices_by_sector.items()
    }
    ticker_sector = {
        t: sector for sector, df in prices_by_sector.items() for t in df.columns
    }

    def _prices_for(pair):
        """Return the sector price frame that contains this pair's tickers."""
        return prices_by_sector[ticker_sector[pair[0]]]

    fold_returns = []
    log_rows = []

    start_idx = 0
    fold = 0
    while start_idx + formation_window + trading_period <= n:
        form_start = start_idx
        form_end   = start_idx + formation_window          # exclusive
        trade_end  = form_end + trading_period             # exclusive

        formation_dates = all_dates[form_start:form_end]
        trading_dates   = all_dates[form_end:trade_end]
        context_dates   = all_dates[form_start:trade_end]

        # --- 1. Select (or use fixed) pairs for this fold ---
        if fixed_pairs is not None:
            selected = pd.DataFrame([
                {"y": y, "x": x, "pair": _pair_name(y, x),
                 "coint_pvalue": np.nan, "hedge_ratio": np.nan, "selected": True}
                for (y, x) in fixed_pairs
            ])
        else:
            per_sector = []
            for sector, df in prices_by_sector.items():
                form_prices = df.loc[df.index.isin(formation_dates)]
                ranked = select_pairs_formation(
                    form_prices,
                    candidate_pairs_by_sector[sector],
                    alpha=alpha,
                    top_n=top_n,
                    unique_legs=unique_legs,
                )
                per_sector.append(ranked)
            ranked_all = (
                pd.concat(per_sector, ignore_index=True)
                if per_sector else pd.DataFrame()
            )
            selected = (
                ranked_all[ranked_all["selected"]]
                if len(ranked_all) else ranked_all
            )

        # --- 2. Trade the selected pairs over the trading window ---
        if len(selected):
            # Group by sector-owning price frame so each pair reads correct prices.
            for _, r in selected.iterrows():
                prices = _prices_for((r["y"], r["x"]))
                one = pd.DataFrame([r])
                fr = _fold_returns(
                    prices, one, context_dates, trading_dates,
                    entry_z, exit_z, obs_cov, cost_per_leg, n_legs,
                )
                if len(fr):
                    fr.name = r["pair"]
                    fold_returns.append(("_fold_%d" % fold, fr))

            for _, r in selected.iterrows():
                log_rows.append({
                    "fold": fold,
                    "formation_start": formation_dates[0],
                    "formation_end": formation_dates[-1],
                    "trading_start": trading_dates[0],
                    "trading_end": trading_dates[-1],
                    "pair": r["pair"],
                    "coint_pvalue": r["coint_pvalue"],
                    "hedge_ratio": r["hedge_ratio"],
                })

        start_idx += trading_period
        fold += 1

    # Aggregate: within each fold equal-weight the pairs, then concat folds.
    if not fold_returns:
        return (
            pd.Series(dtype=float, name="strategy_ret"),
            pd.DataFrame(log_rows),
        )

    fold_frames = {}
    for fold_key, series in fold_returns:
        fold_frames.setdefault(fold_key, []).append(series)

    fold_portfolios = []
    for fold_key, series_list in fold_frames.items():
        fold_df = pd.concat(series_list, axis=1)
        fold_portfolios.append(fold_df.mean(axis=1))  # equal weight this fold

    portfolio_ret = (
        pd.concat(fold_portfolios).sort_index().rename("strategy_ret")
    )
    return portfolio_ret, pd.DataFrame(log_rows)
