import numpy as np
import pandas as pd 

# Calculate returns for sensitivity analysis 
def compute_strategy_returns(df):
    df = df.copy()

    # Spread change
    df["spread_ret"] = df["spread_t"].diff()

    # Strategy return (lag position to avoid lookahead bias)
    df["strategy_ret"] = df["position"].shift(1) * df["spread_ret"]

    return df



# DEAL WITH OUT OF SAMPLE STUFF BEFORE CONTINUING ON WITH THE BACKTEST