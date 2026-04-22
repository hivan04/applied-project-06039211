import numpy as np
import pandas as pd

# Extreme z-score calculation
def flag_extreme_zscore(df, threshold=5):
    """
    Calculates the z-score to determine whether returns are ±5 standard deviations away from the mean.
    We will proceed to winsorize the values if so.
    """
    z_scores = (df - df.mean()) / df.std()
    flags = np.abs(z_scores) > threshold
    return flags

# Winsorize extreme values
def winsorize(df, lower=0.01, upper=0.99):
    return df.clip(
        lower=df.quantile(lower),
        upper=df.quantile(upper),
        axis=1
    )

