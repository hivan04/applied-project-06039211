import os
import pandas as pd
import matplotlib.pyplot as plt
from src.local_config import *


def pair_volatility_plots(
    vol_df,
    output_dir=PROJECT_ROOT / "outputs/volatility",
    vol_col="pair_strategy_vol",
    date_col="date",
    figsize=(12, 5)
):

    df = vol_df.copy()

    required_cols = [date_col, "strategy", "pair", "obs_cov", vol_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    os.makedirs(output_dir, exist_ok=True)

    df[date_col] = pd.to_datetime(df[date_col])

    group_cols = ["strategy", "pair", "obs_cov"]

    for (strategy_name, pair_name, obs_cov), group in df.groupby(group_cols):

        plot_df = group.sort_values(date_col).copy()

        # Skip groups where volatility is fully missing
        if plot_df[vol_col].dropna().empty:
            continue

        plt.figure(figsize=figsize)

        plt.plot(
            plot_df[date_col],
            plot_df[vol_col],
            label="Pair Strategy Volatility"
        )

        plt.title(
            f"Pair Strategy Return Volatility\n"
            f"{pair_name} | {strategy_name} | R={obs_cov}"
        )
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Make filename safe
        safe_strategy = str(strategy_name).replace("/", "_").replace(" ", "_")
        safe_pair = str(pair_name).replace("/", "_").replace(" ", "_")
        safe_obs_cov = str(obs_cov).replace(".", "p")

        filename = f"{safe_strategy}_{safe_pair}_R_{safe_obs_cov}_volatility.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Volatility plots saved to: {output_dir}")