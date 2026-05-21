# Libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from itertools import combinations
import pickle
import os
import sys
from pykalman import KalmanFilter
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)

# Import scripts
from src.backtest import *
from src.cointegration import *
from src.drawdown_optimisation import *
from src.honia_rf import *
from src.kalman import *
from src.local_config import *
from src.pairs import *
from src.plots import *
from src.portfolio_con import *
from src.refined_trading_signal import *
from src.rolling_backtest import *
from src.trading_signal import *
from src.utils import *
from src.volatility_plot import *


"""
-- Table Contents: --
1) Data Cleaning
2) Cointegration Test
3) Pairs Strategy 
4) Portfolio Construction
5) Trading Signal
6) Backtest
7) Refined Backtest
8) Rolling Backtest
"""

# 1) Data Cleaning
"""
df1 = tech assets 
df2 = commodities assets 
df = tech + commodities merged together
"""

df1 = pd.read_csv(PROJECT_ROOT / "data/raw_data/tech.csv")
df2 = pd.read_csv(PROJECT_ROOT / "data/raw_data/commodites.csv")

df = pd.merge(df1, df2, how='left', on='Date')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.set_index('Date')
df = df.sort_index()

# Set the time index for our respective dataframes
df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True)
df1 = df1.set_index('Date')

df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True)
df2 = df2.set_index('Date')

df1 = df1.loc[:'2026-01-01']
last_date = df1.index.max()
df1 = df1.loc[:, df1.loc[last_date].notna()]

df2 = df2.loc[:'2026-01-01']
last_date = df2.index.max()
df2 = df2.loc[:, df2.loc[last_date].notna()]

# Transforming raw prices into log prices
df1 = np.log(df1)
df2 = np.log(df2)

split_date = "2022-10-01"
df1_is = df1.loc[df1.index < split_date].copy()
df1_oos = df1.loc[df1.index >= split_date].copy()

df2_is = df2.loc[df2.index < split_date].copy()
df2_oos = df2.loc[df2.index >= split_date].copy()

tech_returns = (df1 - df1.shift(1)).dropna()
comm_returns = (df2 - df2.shift(1)).dropna()

tech_returns_is = tech_returns.loc[tech_returns.index < split_date].copy()
tech_returns_oos = tech_returns.loc[tech_returns.index >= split_date].copy()

comm_returns_is = comm_returns.loc[comm_returns.index < split_date].copy()
comm_returns_oos = comm_returns.loc[comm_returns.index >= split_date].copy()
comm_returns = comm_returns.drop(["189 HK Equity", "1378 HK Equity"], axis=1)
comm_returns.columns

# 2) Cointegration Test
pairs_t = list(combinations(df1_is.columns, 2))
pairs_t

# Create dictionary for cointegration test and run it for each pair using the for loop below
results_t = {}

for x, y in pairs_t:
    pair_name = f"{x}-{y}"   
    
    res = analyze_pair(
        y_series=df1_is[x],
        x_series=df1_is[y],
        pair_name=pair_name  
    )
    
    results_t[pair_name] = res

summary_t = summarize_results(results_t)

summary_coint_t = summary_t[summary_t["cointegrated"]]

strong_pairs_t = summary_t[
    (summary_t["cointegrated"] == True) & 
    (summary_t["coint_test_pvalue"] < 0.05)]

weaker_pairs_t = summary_t[
    (summary_t["coint_test_pvalue"] < 0.10) &
    (summary_t["coint_test_pvalue"] >= 0.05)
]

# Calculate spread
spread_t1 = compute_spread(df1_is["1347 HK Equity"], 
                           df1_is["268 HK Equity"], 
                           summary_coint_t["hedge_ratio"].iloc[0])

spread_t2 = compute_spread(df1_is["700 HK Equity"], 
                           df1_is["1347 HK Equity"], 
                           weaker_pairs_t["hedge_ratio"].iloc[0])

spread_t3 = compute_spread(df1_is["981 HK Equity"], 
                           df1_is["3888 HK Equity"], 
                           weaker_pairs_t["hedge_ratio"].iloc[1])

spread_dict_t = {
    "1347 vs 268": spread_t1,
    "700 vs 1347": spread_t2,
    "981 vs 3888": spread_t3
}

z_t1 = compute_zscore(spread_t1)
z_t2 = compute_zscore(spread_t2)
z_t3 = compute_zscore(spread_t3)

z_dict_t = {
    "1347 vs 268": z_t1,
    "700 vs 1347": z_t2,
    "981 vs 3888": z_t3
}

# Generate pairs 
pairs_c = list(combinations(df2_is.columns, 2))
pairs_c

# Create dictionary for cointegration test and run it for each pair using the for loop below
results_c = {}

for x, y in pairs_c:
    pair_name = f"{x}-{y}"   
    
    res = analyze_pair(
        y_series=df2_is[x],
        x_series=df2_is[y],
        pair_name=pair_name  
    )
    
    results_c[pair_name] = res

summary_c = summarize_results(results_c)
summary_coint_c = summary_c[summary_c["cointegrated"]]


strong_pairs_c = summary_c[
    (summary_c["cointegrated"] == True) & 
    (summary_c["coint_test_pvalue"] < 0.05)]

weaker_pairs_c = summary_c[
    (summary_c["coint_test_pvalue"] < 0.10) &
    (summary_c["coint_test_pvalue"] >= 0.05)
]

spread_c1 = compute_spread(df2_is["857 HK Equity"], 
                           df2_is["2386 HK Equity"], 
                           summary_coint_c["hedge_ratio"].iloc[0])

spread_c2 = compute_spread(df2_is["3993 HK Equity"], 
                           df2_is["2689 HK Equity"], 
                           summary_coint_c["hedge_ratio"].iloc[1])

spread_c3 = compute_spread(df2_is["1258 HK Equity"], 
                           df2_is["3899 HK Equity"], 
                           summary_coint_c["hedge_ratio"].iloc[2])

spread_c4 = compute_spread(df2_is["1258 HK Equity"], 
                           df2_is["189 HK Equity"], 
                           summary_coint_c["hedge_ratio"].iloc[3])

spread_c5 = compute_spread(df2_is["1378 HK Equity"], 
                           df2_is["2099 HK Equity"], 
                           weaker_pairs_c["hedge_ratio"].iloc[0])

spread_c6 = compute_spread(df2_is["1378 HK Equity"], 
                           df2_is["358 HK Equity"], 
                           weaker_pairs_c["hedge_ratio"].iloc[1])

spread_c7 = compute_spread(df2_is["1088 HK Equity"], 
                           df2_is["1171 HK Equity"], 
                           weaker_pairs_c["hedge_ratio"].iloc[2])

spread_c8 = compute_spread(df2_is["2099 HK Equity"], 
                           df2_is["1164 HK Equity"], 
                           weaker_pairs_c["hedge_ratio"].iloc[3])

spread_dict_c = {
    "857 vs 2386": spread_c1,
    "3993 vs 2689": spread_c2,
    "1258 vs 3899": spread_c3,
    "1258 vs 189": spread_c4,
    "1378 vs 2099": spread_c5,
    "1378 vs 358": spread_c6,
    "1088 vs 1171": spread_c7,
    "2099 vs 1164": spread_c8
}

z_c1 = compute_zscore(spread_c1)
z_c2 = compute_zscore(spread_c2)
z_c3 = compute_zscore(spread_c3)
z_c4 = compute_zscore(spread_c4)
z_c5 = compute_zscore(spread_c5)
z_c6 = compute_zscore(spread_c6)
z_c7 = compute_zscore(spread_c7)
z_c8 = compute_zscore(spread_c8)

z_dict_c = {
    "857 vs 2386": z_c1,
    "3993 vs 2689": z_c2,
    "1258 vs 3899": z_c3,
    "1258 vs 189": z_c4,
    "1378 vs 2099": z_c5,
    "1378 vs 358": z_c6,
    "1088 vs 1171": z_c7,
    "2099 vs 1164": z_c8
}

pairs_rolling = [
    {"y": "1347 HK Equity", "x": "268 HK Equity", "df": df1_is},
    {"y": "700 HK Equity", "x": "1347 HK Equity", "df": df1_is},
    {"y": "981 HK Equity", "x": "3888 HK Equity", "df": df1_is},
    {"y": "857 HK Equity", "x": "2386 HK Equity", "df": df2_is},
    {"y": "3993 HK Equity", "x": "2689 HK Equity", "df": df2_is},
    {"y": "1258 HK Equity", "x": "3899 HK Equity", "df": df2_is},
    {"y": "1258 HK Equity", "x": "189 HK Equity", "df": df2_is},
    {"y": "1378 HK Equity", "x": "2099 HK Equity", "df": df2_is},
    {"y": "1378 HK Equity", "x": "358 HK Equity", "df": df2_is},
    {"y": "1088 HK Equity", "x": "1171 HK Equity", "df": df2_is},
    {"y": "2099 HK Equity", "x": "1164 HK Equity", "df": df2_is},
]

results_rolling = []

for pair in pairs_rolling:
    y_series = pair["df"][pair["y"]]
    x_series = pair["df"][pair["x"]]
    
    result = analyze_rolling_pair(
        y_series,
        x_series,
        pair_name=f"{pair['y']} vs {pair['x']}"
    )
    
    results_rolling.append(result)

rolling_res = summarize_results_list(results_rolling) 

trading_pairs = rolling_res[rolling_res["cointegrated"] == True]
trading_pairs

# 3) Pairs Strategy
pairs_is = [
    {"y": "1347 HK Equity", "x": "268 HK Equity", "df": df1_is},
    {"y": "857 HK Equity", "x": "2386 HK Equity", "df": df2_is},
    {"y": "3993 HK Equity", "x": "2689 HK Equity", "df": df2_is},
    {"y": "1258 HK Equity", "x": "3899 HK Equity", "df": df2_is},
    {"y": "1258 HK Equity", "x": "189 HK Equity", "df": df2_is},
]

# Static
static_summaries_is = []
static_spreads_is = {}
static_models_is = {}

for pair in pairs_is:
    y_series = pair["df"][pair["y"]]
    x_series = pair["df"][pair["x"]]
    pair_name = f"{pair['y']} vs {pair['x']}"

    summary_df, spread, model = estimate_hedge_ratio_df(
        y_series,
        x_series,
        pair_name=pair_name
    )

    static_summaries_is.append(summary_df)
    static_spreads_is[pair_name] = spread
    static_models_is[pair_name] = model

static_results_df_is = pd.concat(static_summaries_is, ignore_index=True)

# Dynamic
dynamic_summaries_is = []
dynamic_spreads_is = {}
dynamic_details_is = {}

obs_cov_values = [0.5, 1.0, 5.0]

for pair in pairs_is:
    y_series = pair["df"][pair["y"]]
    x_series = pair["df"][pair["x"]]
    pair_name = f"{pair['y']} vs {pair['x']}"

    for r in obs_cov_values:
        summary_df, spread, data = estimate_dynamic_hedge_ratio_df(
            y_series,
            x_series,
            pair_name=pair_name,
            obs_cov=r
        )

        summary_df["obs_cov"] = r
        summary_df["pair"] = pair_name

        key = (pair_name, r)

        dynamic_summaries_is.append(summary_df)
        dynamic_spreads_is[key] = spread
        dynamic_details_is[key] = data

dynamic_results_df_is = pd.concat(dynamic_summaries_is, ignore_index=True)
dynamic_results_df_is.sort_values(by=["obs_cov", "pair"]).reset_index(drop=True)

dynamic_results_is = dynamic_results_df_is[dynamic_results_df_is["obs_cov (R)"] == 1.0]

pairs_oos = [
    {"y": "1347 HK Equity", "x": "268 HK Equity",  "df": df1_oos},
    {"y": "857 HK Equity",  "x": "2386 HK Equity", "df": df2_oos},
    {"y": "3993 HK Equity", "x": "2689 HK Equity", "df": df2_oos},
    {"y": "1258 HK Equity", "x": "3899 HK Equity", "df": df2_oos},
]

static_summaries_oos = []
static_spreads_oos = {}
static_models_oos = {}

for pair in pairs_oos:
    y_series = pair["df"][pair["y"]]
    x_series = pair["df"][pair["x"]]
    pair_name = f"{pair['y']} vs {pair['x']}"

    summary_df, spread, model = estimate_hedge_ratio_df(
        y_series,
        x_series,
        pair_name=pair_name
    )

    static_summaries_oos.append(summary_df)
    static_spreads_oos[pair_name] = spread
    static_models_oos[pair_name] = model

static_results_df_oos = pd.concat(static_summaries_oos, ignore_index=True)

dynamic_summaries_oos = []
dynamic_spreads_oos = {}
dynamic_details_oos = {}

obs_cov_values = [0.5, 1.0, 5.0]

for pair in pairs_oos:
    y_series = pair["df"][pair["y"]]
    x_series = pair["df"][pair["x"]]
    pair_name = f"{pair['y']} vs {pair['x']}"

    for r in obs_cov_values:
        summary_df, spread, data = estimate_dynamic_hedge_ratio_df(
            y_series,
            x_series,
            pair_name=pair_name,
            obs_cov=r
        )

        summary_df["obs_cov"] = r
        summary_df["pair"] = pair_name

        key = (pair_name, r)

        dynamic_summaries_oos.append(summary_df)
        dynamic_spreads_oos[key] = spread
        dynamic_details_oos[key] = data

dynamic_results_df_oos = pd.concat(dynamic_summaries_oos, ignore_index=True)
dynamic_results_df_oos.sort_values(by=["obs_cov", "pair"]).reset_index(drop=True)

dynamic_results_oos = dynamic_results_df_oos[dynamic_results_df_oos["obs_cov (R)"] == 1.0]

# 4) Portfolio Construction

spread_returns_dict = {}

for _, row in static_results_df_is.iterrows():
    pair   = row["pair"]
    y_name, x_name = [s.strip() for s in pair.split(" vs ")]
    df     = df1_is if y_name in df1_is.columns else df2_is
    spread = df[y_name] - row["beta"] * df[x_name]
    spread_returns_dict[pair] = spread.diff().dropna()

results = portfolio_diagnostics(spread_returns_dict)
portfolio_is = [
    {"y": "1347 HK Equity", "x": "268 HK Equity", "df": df1_is},
    {"y": "857 HK Equity", "x": "2386 HK Equity", "df": df2_is},
    {"y": "3993 HK Equity", "x": "2689 HK Equity", "df": df2_is},
    {"y": "1258 HK Equity", "x": "3899 HK Equity", "df": df2_is},
]

portfolio_oos = [
    {"y": "1347 HK Equity", "x": "268 HK Equity", "df": df1_oos},
    {"y": "857 HK Equity", "x": "2386 HK Equity", "df": df2_oos},
    {"y": "3993 HK Equity", "x": "2689 HK Equity", "df": df2_oos},
    {"y": "1258 HK Equity", "x": "3899 HK Equity", "df": df2_oos},
]

portfolio = pd.DataFrame([
    {"pair": "1347 HK Equity vs 268 HK Equity",  "sector": "tech",      "weight": 0.50},
    {"pair": "857 HK Equity vs 2386 HK Equity",  "sector": "commodity", "weight": 1/6},
    {"pair": "3993 HK Equity vs 2689 HK Equity", "sector": "commodity", "weight": 1/6},
    {"pair": "1258 HK Equity vs 3899 HK Equity", "sector": "commodity", "weight": 1/6},
])

portfolio = portfolio.merge(
    static_results_df_is[["pair", "beta"]],
    on="pair"
)

portfolio = portfolio.rename(columns={"beta": "hedge_ratio"})

# 5) Trading Signal
portfolio_pairs = set(portfolio["pair"])

entry_values = [1.5, 2.0, 2.5]
exit_values = [0.0, 0.5, 1.0]

# IS Trading Signals
is_results = {}
is_summary_rows = []

for e in entry_values:
    for x in exit_values:
        key = f"entry_{e}_exit_{x}"

        is_signals_dict = generate_kalman_signals(
            dynamic_details_is,
            entry_z=e,
            exit_z=x
        )

        is_results[key] = is_signals_dict

        for (pair_name, R), df in is_signals_dict.items():
            is_summary_rows.append({
                "strategy": key,
                "pair": pair_name,
                "obs_cov": R,
                "entry_z": e,
                "exit_z": x,
                "num_trades": df["position"].diff().abs().sum() / 2,
                "mean_zscore": df["zscore"].mean(),
                "std_zscore": df["zscore"].std()
            })

is_trading_signals = pd.DataFrame(is_summary_rows)

dynamic_details_portfolio_is = {
    k: v for k, v in dynamic_details_is.items()
    if k[0] in portfolio_pairs
}

dynamic_details_portfolio_oos = {
    k: v for k, v in dynamic_details_oos.items()
    if k[0] in portfolio_pairs
}

# OOS Trading Signals
oos_results      = {}
oos_summary_rows = []

for e in entry_values:
    for x in exit_values:
        key = f"entry_{e}_exit_{x}"

        oos_signals_dict = generate_kalman_signals(
            dynamic_details_portfolio_oos,
            entry_z=e,
            exit_z=x
        )

        oos_results[key] = oos_signals_dict

        for (pair_name, R), df in oos_signals_dict.items():
            oos_summary_rows.append({
                "strategy":  key,
                "pair":      pair_name,
                "obs_cov":   R,
                "entry_z":   e,
                "exit_z":    x,
                "num_trades": df["position"].diff().abs().sum() / 2,
                "mean_zscore": df["zscore"].mean(),
                "std_zscore":  df["zscore"].std()
            })

oos_trading_signals = pd.DataFrame(oos_summary_rows)

# 6) Backtest (Baseline Trading Signal)
weights   = portfolio.set_index("pair")["weight"].to_dict()
obs_cov   = 1.0

is_portfolio_dict  = {}
oos_portfolio_dict = {}

for strategy_name, signals_dict in is_results.items():
    pair_rets = {}
    for (pair_name, R), df in signals_dict.items():
        if not np.isclose(float(R), obs_cov) or pair_name not in weights:
            continue
        df = compute_strategy_returns(df.copy())
        pair_rets[pair_name] = df["strategy_ret"] * weights[pair_name]
    if pair_rets:
        is_portfolio_dict[strategy_name] = pd.DataFrame(pair_rets).sum(axis=1)

for strategy_name, signals_dict in oos_results.items():
    pair_rets = {}
    for (pair_name, R), df in signals_dict.items():
        if not np.isclose(float(R), obs_cov) or pair_name not in weights:
            continue
        df = compute_strategy_returns(df.copy())
        pair_rets[pair_name] = df["strategy_ret"] * weights[pair_name]
    if pair_rets:
        oos_portfolio_dict[strategy_name] = pd.DataFrame(pair_rets).sum(axis=1)

is_portfolio_rows = []

for strategy_name, ret in is_portfolio_dict.items():
    ret = ret.dropna()
    mean_ret = ret.mean()
    std_ret  = ret.std()
    sharpe   = mean_ret / std_ret if std_ret != 0 else np.nan

    is_portfolio_rows.append({
        "strategy":    strategy_name,
        "entry_z":     float(strategy_name.split("_")[1]),
        "exit_z":      float(strategy_name.split("_")[3]),
        "sharpe":      sharpe,
        "total_return": ret.sum(),
    })

is_portfolio_signals = pd.DataFrame(is_portfolio_rows)

is_sharpe_pivot = is_portfolio_signals.pivot_table(
    index="entry_z", columns="exit_z", values="sharpe"
)

is_return_pivot = is_portfolio_signals.pivot_table(
    index="entry_z", columns="exit_z", values="total_return"
)

is_metrics  = performance_metrics(is_portfolio_dict["entry_1.5_exit_0.5"],  rf=rf_is)

# OOS Backtest
oos_metrics = performance_metrics(oos_portfolio_dict["entry_1.5_exit_0.5"], rf=rf_oos)

strategy_name = "entry_1.5_exit_0.5"
is_ret  = is_portfolio_dict[strategy_name].fillna(0)
oos_ret = oos_portfolio_dict[strategy_name].fillna(0)


is_cum  = 100 * (1 + is_ret).cumprod()
oos_cum = is_cum.iloc[-1] * (1 + oos_ret).cumprod()

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(is_cum.index,  is_cum,  color="steelblue", lw=1.5, label="In-Sample")
ax.plot(oos_cum.index, oos_cum, color="darkorange", lw=1.5, label="Out-of-Sample")
ax.plot([is_cum.index[-1], oos_cum.index[0]],
        [is_cum.iloc[-1],  oos_cum.iloc[0]], color="grey", lw=0.8)
ax.axvline(is_cum.index[-1], color="black", linestyle="--", lw=1.2, label="IS / OOS split")
ax.axhline(100, color="grey", linestyle=":", lw=0.8)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.yaxis.set_minor_locator(mticker.NullLocator()) 
ax.set_title(f"Portfolio Cumulative PnL  |  {strategy_name}")
ax.set_ylabel("Portfolio Value (Base = 100, log scale)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

is_ret  = is_portfolio_dict[strategy_name].fillna(0)
oos_ret = oos_portfolio_dict[strategy_name].fillna(0)

is_cum  = 100 * (1 + is_ret).cumprod()
oos_cum = 100 * (1 + oos_ret).cumprod()

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

axes[0].plot(is_cum.index, is_cum, color="steelblue", lw=1.5)
axes[0].axhline(100, color="grey", linestyle=":", lw=0.8)
axes[0].set_title(f"In-Sample  |  {strategy_name}")
axes[0].set_ylabel("Portfolio Value (Base = 100, log scale)")
axes[0].set_xlabel("Date")
axes[0].set_yscale("log")
axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
axes[0].yaxis.get_major_formatter().set_scientific(False)
axes[0].xaxis.set_major_locator(mdates.YearLocator())
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0].grid(True, alpha=0.3)

axes[1].plot(oos_cum.index, oos_cum, color="darkorange", lw=1.5)
axes[1].axhline(100, color="grey", linestyle=":", lw=0.8)
axes[1].set_title(f"Out-of-Sample  |  {strategy_name}")
axes[1].set_ylabel("Portfolio Value (Base = 100, log scale)")
axes[1].set_xlabel("Date")
axes[1].set_yscale("log")
axes[1].yaxis.set_major_formatter(mticker.ScalarFormatter())
axes[1].yaxis.get_major_formatter().set_scientific(False)
axes[1].xaxis.set_major_locator(mdates.YearLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
axes[1].yaxis.set_minor_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7) Refined Backtest
is_extraction = extractor(is_results)
oos_extraction = extractor(oos_results)

is_volatility = pair_volatility(is_extraction)
is_drawdown_values = drawdown_thresholds(is_volatility)

oos_volatility = pair_volatility(oos_extraction)
oos_drawdown_values = drawdown_thresholds(oos_volatility)

# Portfolio Drawdown Values
# In-Sample
strategy_name = "entry_1.5_exit_0.5"

drawdown_thresholds_is = is_drawdown_values[
    is_drawdown_values["strategy"] == strategy_name
][["date", "pair", "obs_cov", "reduce_drawdown_threshold", "close_drawdown_threshold"]]

# Out-of-Sample
drawdown_thresholds_oos = oos_drawdown_values[
    oos_drawdown_values["strategy"] == strategy_name
][["date", "pair", "obs_cov", "reduce_drawdown_threshold", "close_drawdown_threshold"]]

weights = portfolio.set_index("pair")["weight"].to_dict()
portfolio_pairs = set(portfolio["pair"])

dynamic_details_portfolio_is = {
    k: v for k, v in dynamic_details_is.items()
    if k[0] in portfolio_pairs
}

dynamic_details_portfolio_oos = {
    k: v for k, v in dynamic_details_oos.items()
    if k[0] in portfolio_pairs
}

is_signals_df  = generate_refined_kalman_signals_1(
    dynamic_details     = dynamic_details_portfolio_is,
    drawdown_thresholds = drawdown_thresholds_is   
)

oos_signals_df = generate_refined_kalman_signals_1(
    dynamic_details     = dynamic_details_portfolio_oos,
    drawdown_thresholds = drawdown_thresholds_oos
)

portfolio  = pd.read_csv(PROJECT_ROOT / "data/portfolio")
weights    = portfolio.set_index("pair")["weight"].to_dict()
obs_cov    = 1.0

def aggregate_refined_portfolio(signals_df, weights, obs_cov):
    filtered = signals_df[
        (np.isclose(signals_df["obs_cov"], obs_cov)) &
        (signals_df["pair"].isin(weights))
    ].copy()

    filtered["spread_change"] = filtered.groupby("pair")["spread_t"].diff()
    filtered["strategy_ret"]  = (
        filtered["active_position"]
        * filtered["spread_change"]
        * filtered["pair"].map(weights)
    )

    return filtered.groupby(filtered.index)["strategy_ret"].sum()

is_portfolio_ret  = aggregate_refined_portfolio(is_signals_df,  weights, obs_cov)
oos_portfolio_ret = aggregate_refined_portfolio(oos_signals_df, weights, obs_cov)

is_refined_metrics  = performance_metrics(is_portfolio_ret,  rf=rf_is)
oos_refined_metrics = performance_metrics(oos_portfolio_ret, rf=rf_oos)

# Refined Trading Signal w/o Drawdown Parameter
is_signals_2_df = generate_refined_kalman_signals_2(
    dynamic_details_portfolio_is,
    use_fixed_bands=False,
    upper_quantile=0.80,
    lower_quantile=0.20
)

oos_signals_2_df = generate_refined_kalman_signals_2(
    dynamic_details = dynamic_details_portfolio_oos,
    use_fixed_bands=False,
    upper_quantile=0.80,
    lower_quantile=0.20
)

is_portfolio_ret_2  = aggregate_refined_portfolio(is_signals_2_df,  weights, obs_cov)
oos_portfolio_ret_2 = aggregate_refined_portfolio(oos_signals_2_df, weights, obs_cov)

is_refined_metrics_2  = performance_metrics(is_portfolio_ret_2,  rf=rf_is)
oos_refined_metrics_2 = performance_metrics(oos_portfolio_ret_2, rf=rf_oos)

def build_portfolio_ret(results, strategy_name, weights, obs_cov):
    pair_rets = {}
    for (pair_name, R), df in results[strategy_name].items():
        if not np.isclose(float(R), obs_cov) or pair_name not in weights:
            continue
        df = compute_strategy_returns(df.copy())
        pair_rets[pair_name] = df["strategy_ret"] * weights[pair_name]
    return pd.DataFrame(pair_rets).sum(axis=1)

is_baseline  = build_portfolio_ret(is_results,  strategy_name, weights, obs_cov)
oos_baseline = build_portfolio_ret(oos_results, strategy_name, weights, obs_cov)

is_cum_base  = 100 * (1 + is_baseline.fillna(0)).cumprod()
oos_cum_base = 100 * (1 + oos_baseline.fillna(0)).cumprod()
is_cum_ref   = 100 * (1 + is_portfolio_ret.fillna(0)).cumprod()
oos_cum_ref  = 100 * (1 + oos_portfolio_ret.fillna(0)).cumprod()

oos_cum_base_cont = is_cum_base.iloc[-1] * (1 + oos_baseline.fillna(0)).cumprod()
oos_cum_ref_cont  = is_cum_ref.iloc[-1]  * (1 + oos_portfolio_ret.fillna(0)).cumprod()

fig, ax = plt.subplots(figsize=(14, 5))
for is_cum, oos_cum, color, label in [
    (is_cum_base, oos_cum_base_cont, "steelblue",  "Baseline"),
    (is_cum_ref,  oos_cum_ref_cont,  "darkorange", "Refined (dynamic bands)"),
]:
    ax.plot(is_cum.index,  is_cum,  color=color, lw=1.5, label=label)
    ax.plot(oos_cum.index, oos_cum, color=color, lw=1.5)
    ax.plot([is_cum.index[-1], oos_cum.index[0]],
            [is_cum.iloc[-1],  oos_cum.iloc[0]], color="grey", lw=0.8)

ax.axvline(is_cum_base.index[-1], color="black", linestyle="--", lw=1.2, label="IS / OOS split")
ax.axhline(100, color="grey", linestyle=":", lw=0.8)
ax.set_yscale("log")
ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_title(f"Portfolio Cumulative PnL  |  {strategy_name}  |  Baseline vs Refined (log scale)")
ax.set_ylabel("Portfolio Value (Base = 100, log scale)")
ax.set_xlabel("Date")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(save_dir / f"Combined_{strategy_name}_baseline_vs_refined.png", dpi=300, bbox_inches="tight")
plt.show()

# 8) Rolling Backtest
dynamic_details_full = {}
for key in dynamic_details_oos:
    is_df  = dynamic_details_is.get(key, pd.DataFrame())
    oos_df = dynamic_details_oos[key]
    dynamic_details_full[key] = pd.concat([is_df, oos_df]).sort_index()

# Filter to portfolio pairs
portfolio       = pd.read_csv(PROJECT_ROOT / "data/portfolio")
weights         = portfolio.set_index("pair")["weight"].to_dict()
portfolio_pairs = set(portfolio["pair"])

dynamic_details_full = {k: v for k, v in dynamic_details_full.items() if k[0] in portfolio_pairs}

# Combined risk-free rate
rf_is   = pd.read_csv(PROJECT_ROOT / "data/rf_is",  index_col=0, parse_dates=True)
rf_oos  = pd.read_csv(PROJECT_ROOT / "data/rf_oos", index_col=0, parse_dates=True)
rf_full = pd.concat([rf_is, rf_oos]).sort_index()

print("Pairs:", sorted(set(k[0] for k in dynamic_details_full)))
ref = dynamic_details_full[next(iter(dynamic_details_full))]
print("Date range:", ref.index[0].date(), "→", ref.index[-1].date())

entry_values = [1.0, 1.5, 2.0]
exit_values  = [0.0, 0.5, 1.0]

rolling_results = {}

for e in entry_values:
    for x in exit_values:
        key = f"entry_{e}_exit_{x}"
        rolling_results[key] = walk_forward_backtest(
            dynamic_details = dynamic_details_full,
            weights         = weights,
            entry_z         = e,
            exit_z          = x,
            obs_cov         = 1.0,
            is_window       = 262,
            oos_step        = 63,
        )

print("Strategies computed:", list(rolling_results.keys()))

# Performance metrics for all strategies
metrics_rows = []
for key, ret in rolling_results.items():
    rf_aligned = rf_full.reindex(ret.index).ffill()
    m = performance_metrics(ret, rf=rf_aligned)
    m["strategy"] = key
    metrics_rows.append(m)

rolling_metrics_df = pd.DataFrame(metrics_rows).set_index("strategy")
rolling_metrics_df

print("Completed!")