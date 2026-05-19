import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF Helper
def _compute_vif(returns_df):
    return pd.Series(
        {col: variance_inflation_factor(returns_df.values, i)
         for i, col in enumerate(returns_df.columns)},
        name="VIF"
    )


def _drop_high_correlation(returns_df, corr_threshold=0.7, coint_pvalues=None):
    retained = returns_df.copy()
    dropped = []

    while len(retained.columns) > 1:
        corr = retained.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        if upper.max().max() <= corr_threshold:
            break

        # All pairs involved in any correlation above the threshold
        exceeds = (upper > corr_threshold).any(axis=0) | (upper > corr_threshold).any(axis=1)
        candidates = retained.columns[exceeds].tolist()

        # Drop the most redundant: highest average absolute correlation to all others.
        # Tiebreak by coint_pvalue — weaker cointegration (higher p-value) goes first.
        if coint_pvalues is not None:
            worst = max(candidates, key=lambda p: (
                corr[p].drop(p).mean(),
                coint_pvalues.get(p, 0)
            ))
        else:
            worst = max(candidates, key=lambda p: corr[p].drop(p).mean())

        dropped.append((worst, round(upper[worst].max() if worst in upper.columns else corr[worst].drop(worst).max(), 3)))
        retained = retained.drop(columns=[worst])

    return retained, dropped


def portfolio_diagnostics(spread_returns_dict, coint_pvalues=None, vif_threshold=10, corr_threshold=0.7):
    """
    Run correlation and VIF diagnostics on a set of pair spread returns,
    iteratively dropping the most collinear pair until all VIF < vif_threshold.

    Parameters
    ----------
    spread_returns_dict : dict
        {pair_name: pd.Series of daily spread returns (strategy_ret)}
    coint_pvalues : dict, optional
        {pair_name: float} cointegration p-value for each pair.
        Used as a tiebreaker when dropping — weaker pairs (higher p-value)
        are dropped first when VIF values are close.
    vif_threshold : float
        Maximum acceptable VIF. Default 10.

    Returns
    -------
    dict with keys:
        correlation    : pd.DataFrame — full correlation matrix of all pairs
        initial_vif    : pd.Series   — VIF before any dropping
        final_vif      : pd.Series   — VIF of retained pairs
        retained_pairs : list        — pairs kept after pruning
        dropped_pairs  : list of (pair_name, vif_at_drop) tuples
    """
    returns_df = pd.DataFrame(spread_returns_dict).dropna()

    if returns_df.shape[1] < 2:
        raise ValueError("Need at least 2 pairs to run diagnostics.")

    corr = returns_df.corr()
    initial_vif = _compute_vif(returns_df)

    # Step 1: drop pairs with pairwise correlation > corr_threshold
    retained, corr_dropped = _drop_high_correlation(returns_df, corr_threshold, coint_pvalues)

    # Step 2: drop remaining pairs with VIF > vif_threshold
    vif_dropped = []

    while len(retained.columns) > 1:
        vif = _compute_vif(retained)

        if vif.max() <= vif_threshold:
            break

        candidates = vif[vif > vif_threshold]

        if coint_pvalues is not None:
            worst = max(
                candidates.index,
                key=lambda p: (candidates[p], coint_pvalues.get(p, 0))
            )
        else:
            worst = candidates.idxmax()

        vif_dropped.append((worst, round(vif[worst], 2)))
        retained = retained.drop(columns=[worst])

    final_vif = _compute_vif(retained) if len(retained.columns) > 1 else pd.Series(
        {retained.columns[0]: 1.0}, name="VIF"
    )

    return {
        "correlation":    corr,
        "initial_vif":    initial_vif,
        "final_vif":      final_vif,
        "retained_pairs": list(retained.columns),
        "corr_dropped":   corr_dropped,
        "vif_dropped":    vif_dropped,
    }


def net_stock_exposures(results_dict, strategy_name, obs_cov, hedge_ratios):
    """
    Compute net notional exposure to each underlying stock across all retained pairs,
    based on current positions.

    Parameters
    ----------
    results_dict : dict
        {strategy_name: {(pair_name, R): signal_df}} — from is_results or oos_results.
    strategy_name : str
        Strategy key, e.g. 'entry_2.0_exit_0.0'.
    obs_cov : float
        Kalman observation covariance R.
    hedge_ratios : dict
        {pair_name: float} hedge ratio (beta) for each pair.

    Returns
    -------
    pd.DataFrame
        Columns: date index, rows: stock tickers.
        Values: net position (positive = net long, negative = net short).
    """
    signals = results_dict[strategy_name]
    exposure_dict = {}

    for (pair_name, R), df in signals.items():
        if not np.isclose(float(R), obs_cov):
            continue

        parts = pair_name.split(" vs ")
        if len(parts) != 2:
            continue

        y_name, x_name = parts[0].strip(), parts[1].strip()
        beta = hedge_ratios.get(pair_name, 1.0)
        pos  = df["position"]

        # Long spread: +1 unit y, -beta units x
        # Short spread: -1 unit y, +beta units x
        exposure_dict[(pair_name, y_name)] = pos * 1.0
        exposure_dict[(pair_name, x_name)] = pos * -beta

    if not exposure_dict:
        return pd.DataFrame()

    rows = {}
    for (pair, stock), series in exposure_dict.items():
        rows[stock] = rows.get(stock, 0) + series

    return pd.DataFrame(rows).fillna(0)
