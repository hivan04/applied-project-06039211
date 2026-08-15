"""
config.py — Project-wide, machine-independent settings (tracked in git).

Machine-specific values (e.g. PROJECT_ROOT) live in the gitignored
local_config.py. Shared assumptions that affect results — such as transaction
costs, the Kalman observation covariance, the z-score bands and the
walk-forward windows — belong here so a single edit re-runs every backtest
consistently.
"""

# --- Transaction cost assumptions ---
# Cost charged per leg, per unit of position turnover, expressed as a fraction
# (0.0015 = 15 basis points). A pairs trade transacts TRANSACTION_LEGS legs
# (long one name, short the other), so a full entry costs
# TRANSACTION_LEGS * TRANSACTION_COST_PER_LEG and a position flip costs double.
TRANSACTION_COST_PER_LEG = 0.0015   # 15 bps per leg
TRANSACTION_LEGS         = 2        # legs per pairs trade

# --- Signal / strategy configuration ---
OBS_COV = 1.0    # Kalman observation covariance R used throughout
ENTRY_Z = 1.5    # z-score entry threshold for the headline strategy
EXIT_Z  = 0.0    # z-score exit threshold for the headline strategy

# --- Walk-forward (rolling) backtest windows, in trading days ---
IS_WINDOW = 262  # in-sample context window per fold (~1 year)
OOS_STEP  = 63   # out-of-sample step harvested per fold (~1 quarter)

# --- Metrics ---
ANNUALISATION_FACTOR = 252  # trading days per year
ROLLING_WINDOW       = 252  # window for the rolling annualised Sharpe (~1 year)

# --- Bootstrap (Sharpe confidence intervals) ---
N_BOOTSTRAP    = 1000  # number of bootstrap replicates
BOOTSTRAP_SEED = 42    # seed for reproducibility
