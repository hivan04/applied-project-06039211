import numpy as np
import matplotlib.pyplot as plt


def bootstrap_sharpe_ci(
    returns,
    n_bootstrap=1000,
    block_size=None,
    ci=0.95,
    annualisation_factor=252,
    rf=None,
    seed=None,
):
    """
    Block bootstrap confidence interval for the annualised Sharpe ratio.

    Uses circular block bootstrap to preserve autocorrelation structure
    in daily return series — resamples blocks of consecutive days rather
    than individual observations.

    Parameters
    ----------
    returns : pd.Series or array-like
        Daily return series (proportional returns, not cumulative PnL).
        NaNs are dropped before computing.
    n_bootstrap : int
        Number of bootstrap replicates. Default 1000.
    block_size : int, optional
        Block length in trading days. Defaults to max(5, int(T ** (1/3))),
        a standard heuristic for financial daily returns (~10 days for T=252).
    ci : float
        Confidence level, e.g. 0.95 for a 95% CI. Default 0.95.
    annualisation_factor : int
        Trading days per year. Default 252.
    rf : float, array-like, or pd.Series, optional
        Daily risk-free rate. If a scalar annualised rate is passed (e.g. 0.05),
        divide by annualisation_factor before passing in.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        sharpe            : float      — point estimate of annualised Sharpe
        ci_lower          : float      — lower bound of CI
        ci_upper          : float      — upper bound of CI
        ci_level          : float      — confidence level used
        block_size        : int        — block size used
        n_obs             : int        — number of non-NaN observations
        bootstrap_sharpes : np.ndarray — all n_bootstrap Sharpe values
    """
    rng = np.random.default_rng(seed)

    ret = np.asarray(returns, dtype=float)
    ret = ret[~np.isnan(ret)]
    n = len(ret)

    if n < 10:
        raise ValueError(f"Too few observations ({n}) to bootstrap.")

    if rf is not None:
        rf_arr = np.asarray(rf, dtype=float)
        if rf_arr.ndim == 0:
            ret = ret - float(rf_arr)
        else:
            rf_arr = rf_arr[~np.isnan(rf_arr)][:n]
            ret = ret[:len(rf_arr)] - rf_arr

    if block_size is None:
        block_size = max(5, int(n ** (1 / 3)))

    point_sharpe = _annualised_sharpe(ret, annualisation_factor)

    bootstrap_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = _circular_block_sample(ret, block_size, n, rng)
        bootstrap_sharpes[i] = _annualised_sharpe(sample, annualisation_factor)

    alpha = 1 - ci
    ci_lower = float(np.percentile(bootstrap_sharpes, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2)))

    return {
        "sharpe":            round(point_sharpe, 4),
        "ci_lower":          round(ci_lower, 4),
        "ci_upper":          round(ci_upper, 4),
        "ci_level":          ci,
        "block_size":        block_size,
        "n_obs":             n,
        "bootstrap_sharpes": bootstrap_sharpes,
    }


def _annualised_sharpe(ret, annualisation_factor=252):
    std = ret.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return (ret.mean() / std) * np.sqrt(annualisation_factor)


def _circular_block_sample(ret, block_size, n, rng):
    """Circular block bootstrap: starting positions wrap around the series end."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    indices = np.concatenate([
        np.arange(s, s + block_size) % n for s in starts
    ])
    return ret[indices[:n]]


def plot_sharpe_ci(ci_result, label="Strategy", ax=None, color="steelblue"):
    """
    Histogram of bootstrap Sharpe distribution with CI bounds marked.

    Parameters
    ----------
    ci_result : dict
        Output of bootstrap_sharpe_ci.
    label : str
        Title label for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates a new figure if None.
    color : str
        Histogram colour.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    sharpes = ci_result["bootstrap_sharpes"]
    ci_pct = int(ci_result["ci_level"] * 100)

    ax.hist(sharpes, bins=50, alpha=0.7, color=color, edgecolor="white")
    ax.axvline(
        ci_result["sharpe"], color="black", lw=2,
        label=f"Sharpe = {ci_result['sharpe']:.2f}"
    )
    ax.axvline(
        ci_result["ci_lower"], color="crimson", lw=1.5, ls="--",
        label=f"{ci_pct}% CI  [{ci_result['ci_lower']:.2f}, {ci_result['ci_upper']:.2f}]"
    )
    ax.axvline(ci_result["ci_upper"], color="crimson", lw=1.5, ls="--")
    ax.axvline(0, color="grey", lw=0.8, ls=":")

    ax.set_title(f"Bootstrap Sharpe CI — {label}")
    ax.set_xlabel("Annualised Sharpe")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return ax


def compare_sharpe_cis(results_dict, figsize=(9, 4), title="Sharpe Ratio — Bootstrap CIs"):
    """
    Horizontal error-bar chart comparing Sharpe CIs across multiple strategies or periods.

    Parameters
    ----------
    results_dict : dict
        {label: ci_result_dict} mapping. ci_result_dict is the output of
        bootstrap_sharpe_ci. Order is preserved (use an OrderedDict or Python 3.7+ dict).
    figsize : tuple
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(results_dict.keys())
    sharpes = [results_dict[l]["sharpe"] for l in labels]
    lowers  = [results_dict[l]["sharpe"] - results_dict[l]["ci_lower"] for l in labels]
    uppers  = [results_dict[l]["ci_upper"] - results_dict[l]["sharpe"] for l in labels]
    ci_pct  = int(list(results_dict.values())[0]["ci_level"] * 100)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(labels))
    colors = ["steelblue" if s >= 0 else "crimson" for s in sharpes]

    ax.barh(y_pos, sharpes, xerr=[lowers, uppers], align="center",
            color=colors, alpha=0.75, ecolor="black", capsize=5, height=0.5)
    ax.axvline(0, color="black", lw=0.8, ls="--")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Annualised Sharpe Ratio")
    ax.set_title(f"{title}\n({ci_pct}% confidence intervals, block bootstrap)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    return fig
