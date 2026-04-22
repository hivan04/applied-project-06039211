import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_spread(y_series, x_series, hedge_ratio, intercept=0.0):
    """
    Compute spread from two aligned price series.

    Spread_t = y_t - (intercept + hedge_ratio * x_t)

    Parameters
    ----------
    y_series : pandas Series
        Dependent variable series.
    x_series : pandas Series
        Independent variable series.
    hedge_ratio : float
        Estimated beta from OLS.
    intercept : float, default 0.0
        Estimated intercept from OLS.

    Returns
    -------
    pandas Series
        Spread series aligned on common dates.
    """
    df = pd.concat(
        [
            pd.Series(y_series).rename("y"),
            pd.Series(x_series).rename("x")
        ],
        axis=1
    ).dropna()

    spread = df["y"] - (intercept + hedge_ratio * df["x"])
    spread.name = "spread"
    return spread

# --- #

def compute_zscore(series, window=None):
    """
    Compute z-score of a series.

    Parameters
    ----------
    series : pandas Series
        Input series, usually the spread.
    window : int or None, default None
        If None, use full-sample mean/std.
        If int, use rolling mean/std.

    Returns
    -------
    pandas Series
        Z-score series.
    """
    series = pd.Series(series).dropna()

    if window is None:
        mean = series.mean()
        std = series.std()
    else:
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()

    zscore = (series - mean) / std
    zscore.name = "zscore"
    return zscore

# Plots
def plot_multiple_spreads(spread_dict, figsize=(10, 12)):
    """
    Plot multiple spread time series stacked vertically.

    Parameters
    ----------
    spread_dict : dict
        Dictionary of {pair_name: spread_series}
    color : str, default "black"
        Line color for all plots
    figsize : tuple, default (10, 12)
        Figure size
    """
    n = len(spread_dict)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)

    # Handle single plot case
    if n == 1:
        axes = [axes]

    for ax, (pair_name, spread) in zip(axes, spread_dict.items()):
        sns.lineplot(ax=ax, x=spread.index, y=spread)
        ax.set_title(f"{pair_name} Spread")
        ax.set_ylabel("Spread")
        ax.grid(True)

        # Mean line (important for cointegration)
        ax.axhline(spread.mean(), color="grey", linestyle="--", label="Mean")

    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()

# --- #

def plot_multiple_zscores(zscore_dict, color="black", figsize=(10, 12)):
    """
    Plot multiple z-score time series stacked vertically.

    Parameters
    ----------
    zscore_dict : dict
        Dictionary of {pair_name: zscore_series}
    color : str, default "black"
        Line color for all plots
    figsize : tuple, default (10, 12)
        Figure size
    """
    n = len(zscore_dict)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)

    # If only one plot, make axes iterable
    if n == 1:
        axes = [axes]

    for ax, (pair_name, z) in zip(axes, zscore_dict.items()):
        sns.lineplot(ax=ax, x=z.index, y=z, color=color)
        ax.set_title(f"{pair_name} Z-Score Time Series")
        ax.set_ylabel("Z-Score")
        ax.grid(True)

        # Optional: add thresholds
        ax.axhline(0, color="grey", linestyle="-")
        ax.axhline(2, color="grey", linestyle="--")
        ax.axhline(-2, color="grey", linestyle="--")

    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()