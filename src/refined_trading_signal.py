import numpy as np
import pandas as pd


def generate_refined_kalman_signals_1(
    dynamic_details,
    z_window=60,
    band_window=126,
    upper_quantile=0.80,
    lower_quantile=0.20,
    exit_z=0.5,
    drawdown_thresholds=None,
    reduced_exposure=0.5,
    shift_bands=True,
    use_fixed_bands=False,
    fixed_upper=1.5,
    fixed_lower=-1.5,
):
    all_results = []
    if drawdown_thresholds is not None:
        threshold_df = drawdown_thresholds.copy()
        threshold_df["date"] = pd.to_datetime(threshold_df["date"])
        threshold_df = threshold_df[
            ["date", "pair", "obs_cov",
             "reduce_drawdown_threshold", "close_drawdown_threshold"]
        ].copy()
    else:
        threshold_df = None

    for key, data in dynamic_details.items():
        pair_name, obs_cov = key
        df = data.copy()

        df["spread_t"] = df["y"] - df["alpha_t"] - df["beta_t"] * df["x"]

        df["rolling_mean"] = df["spread_t"].rolling(z_window).mean()
        df["rolling_std"]  = df["spread_t"].rolling(z_window).std()
        df["zscore"]       = (df["spread_t"] - df["rolling_mean"]) / df["rolling_std"]

        if use_fixed_bands:
            df["upper_band"] = fixed_upper
            df["lower_band"] = fixed_lower
        else:
            df["upper_band"] = df["zscore"].rolling(band_window).quantile(upper_quantile)
            df["lower_band"] = df["zscore"].rolling(band_window).quantile(lower_quantile)
            if shift_bands:
                df["upper_band"] = df["upper_band"].shift(1)
                df["lower_band"] = df["lower_band"].shift(1)

        df = df.dropna(subset=["zscore", "upper_band", "lower_band"]).copy()

        df["pair"]    = pair_name
        df["obs_cov"] = obs_cov

        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"])

        if threshold_df is not None:
            df = df.merge(threshold_df, on=["date", "pair", "obs_cov"], how="left")
        else:
            df["reduce_drawdown_threshold"] = np.nan
            df["close_drawdown_threshold"]  = np.nan

        df["raw_position"] = np.nan
        df.loc[df["zscore"] > df["upper_band"], "raw_position"] = -1
        df.loc[df["zscore"] < df["lower_band"], "raw_position"] = 1
        df.loc[df["zscore"].abs() < exit_z,     "raw_position"] = 0
        df["raw_position"] = df["raw_position"].ffill().fillna(0)

        spread_arr     = df["spread_t"].values
        raw_pos_arr    = df["raw_position"].values
        reduce_dd_arr  = df["reduce_drawdown_threshold"].values
        close_dd_arr   = df["close_drawdown_threshold"].values
        n = len(df)

        position_out        = np.zeros(n)
        active_position_out = np.zeros(n)
        trade_pnl_out       = np.zeros(n)
        trade_equity_out    = np.zeros(n)
        running_peak_out    = np.zeros(n)
        drawdown_out        = np.zeros(n)
        risk_state_out      = [""] * n

        current_position     = 0.0
        current_trade_equity = 0.0
        current_peak         = 0.0

        for i in range(n):
            raw_pos      = raw_pos_arr[i]
            spread_now   = spread_arr[i]
            reduce_dd_t  = reduce_dd_arr[i]
            close_dd_t   = close_dd_arr[i]
            active_position_out[i] = current_position

            pnl = 0.0 if i == 0 else current_position * (spread_now - spread_arr[i - 1])
            trade_pnl_out[i] = pnl

            if raw_pos == 0:
                current_position = 0.0; current_trade_equity = 0.0; current_peak = 0.0
                dd = 0.0; state = "flat"
            elif np.sign(raw_pos) != np.sign(current_position) and current_position != 0:
                current_position = raw_pos; current_trade_equity = 0.0; current_peak = 0.0
                dd = 0.0; state = "new_trade"
            else:
                current_trade_equity += pnl
                current_peak = max(current_peak, current_trade_equity)
                dd = current_trade_equity - current_peak
                if pd.notna(close_dd_t) and dd <= -abs(close_dd_t):
                    current_position = 0.0; current_trade_equity = 0.0; current_peak = 0.0
                    state = "closed_by_drawdown"
                elif pd.notna(reduce_dd_t) and dd <= -abs(reduce_dd_t):
                    current_position = reduced_exposure * raw_pos
                    state = "reduced_by_drawdown"
                else:
                    current_position = raw_pos; state = "normal"

            position_out[i]     = current_position
            trade_equity_out[i] = current_trade_equity
            running_peak_out[i] = current_peak
            drawdown_out[i]     = dd
            risk_state_out[i]   = state

        df["position"]        = position_out
        df["active_position"] = active_position_out
        df["trade_pnl"]       = trade_pnl_out
        df["trade_equity"]    = trade_equity_out
        df["running_peak"]    = running_peak_out
        df["trade_drawdown"]  = drawdown_out
        df["risk_state"]      = risk_state_out

        df["z_window"]        = z_window
        df["band_window"]     = band_window
        df["upper_quantile"]  = upper_quantile
        df["lower_quantile"]  = lower_quantile
        df["exit_z"]          = exit_z
        df["reduced_exposure"]= reduced_exposure

        df = df.set_index("date")
        all_results.append(df)

    return pd.concat(all_results, axis=0)


def generate_refined_kalman_signals_2(
    dynamic_details,
    z_window=60,
    band_window=126,
    upper_quantile=0.80,
    lower_quantile=0.20,
    exit_z=0.5,
    shift_bands=True,
    use_fixed_bands=False,
    fixed_upper=1.5,
    fixed_lower=-1.5,
):
    
    all_results = []

    for key, data in dynamic_details.items():
        pair_name, obs_cov = key
        df = data.copy()

        # Kalman spread
        df["spread_t"] = df["y"] - df["alpha_t"] - df["beta_t"] * df["x"]

        # Rolling z-score (dynamic — adapts every bar)
        df["rolling_mean"] = df["spread_t"].rolling(z_window).mean()
        df["rolling_std"]  = df["spread_t"].rolling(z_window).std()
        df["zscore"]       = (df["spread_t"] - df["rolling_mean"]) / df["rolling_std"]

        # Entry bands
        if use_fixed_bands:
            df["upper_band"] = fixed_upper
            df["lower_band"] = fixed_lower
        else:
            df["upper_band"] = (
                df["zscore"].rolling(band_window).quantile(upper_quantile)
            )
            df["lower_band"] = (
                df["zscore"].rolling(band_window).quantile(lower_quantile)
            )
            if shift_bands:
                df["upper_band"] = df["upper_band"].shift(1)
                df["lower_band"] = df["lower_band"].shift(1)

        # Defer dropna until after all rolling calcs to minimise obs loss
        df = df.dropna(subset=["zscore", "upper_band", "lower_band"]).copy()

        # Identifiers
        df["pair"]    = pair_name
        df["obs_cov"] = obs_cov

        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"])

        # Signal generation
        entry_long  = df["zscore"] < df["lower_band"]
        entry_short = df["zscore"] > df["upper_band"]
        exit_sig    = (df["zscore"].abs() < exit_z) & ~entry_long & ~entry_short

        df["position"] = np.nan
        df.loc[entry_short, "position"] = -1
        df.loc[entry_long,  "position"] =  1
        df.loc[exit_sig,    "position"] =  0

        df["position"] = df["position"].ffill().fillna(0)
        df["active_position"] = df["position"].shift(1).fillna(0)

        # P&L
        df["spread_change"] = df["spread_t"].diff()
        df["trade_pnl"]     = df["active_position"] * df["spread_change"]

        df["z_window"]       = z_window
        df["band_window"]    = band_window
        df["upper_quantile"] = upper_quantile
        df["lower_quantile"] = lower_quantile
        df["exit_z"]         = exit_z

        df = df.set_index("date")
        all_results.append(df)

    return pd.concat(all_results, axis=0)