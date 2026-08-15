# Statistical Arbitrage using Co-Integration in HK Tech & Commodities

The main question of this project is to see whether statistical arbitrage works in the Hong Kong sectors: Technology and Commodities.

# Contents

- [Statistical Arbitrage using Co-Integration in HK Tech \& Commodities](#statistical-arbitrage-using-co-integration-in-hk-tech--commodities)
- [Contents](#contents)
- [Key Results from Backtesting](#key-results-from-backtesting)
  - [Baseline Backtest](#baseline-backtest)
  - [Refined Backtest](#refined-backtest)
  - [Rolling-Window Backtest](#rolling-window-backtest)
    - [Sensitivity of Rolling-Window Backtest](#sensitivity-of-rolling-window-backtest)
- [Running project on your own device](#running-project-on-your-own-device)
- [Outline of Pipeline](#outline-of-pipeline)

# Key Results from Backtesting

In this project, 3 backtests were ran (*all three backtests were run with R = 1, where R is the observation noise covariance in the Kalman Filter. This value was chosen as a neutral baseline, reflecting equal confidence in observed price updates and the model's prior estimates.*):

1. Baseline Backtest - this encompasses a static z-score entry/exit and trades based off primarily the Kalman Filter
2. Refined Backtest - this builds off the baseline strategy by including different layers of risk management to the signal
   a) The first refinemnent to the trading signal was implementing a dyanmic z-score band, the same as the baseline but the threshold for entry/exit adjusts accordingly depending on the time-series (of course with no look-ahead bias);
   b) The second refinement was calculating the dynamic maximum drawdown (via. estimated volatility)
3. Rolling Window Backtest (12-Month Window) - backtested both the baseline and refined trading signals using rolling window analysis.

## Baseline Backtest

![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/trading_signal/cumulative_pnl_base_entry_1.5_exit_0.5_baseline_vs_refined.png)

## Refined Backtest

![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/refined_trading_signal/Combined_entry_1.5_exit_0.5_baseline_vs_refined.png)

## Rolling-Window Backtest

![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/rolling_outputs/WalkForward_entry_1.5_exit_0.5_cumulative_pnl.png)
![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/rolling_outputs/WalkForward_entry_1.5_exit_0.5_drawdown_and_sharpe.png)

### Sensitivity of Rolling-Window Backtest

There was a further backtest on the different combination of z-score bands to see the fluctuation in results which can be seen in the plots below:

![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/rolling_outputs/WalkForward_All_Strategies_cumulative_pnl.png)
![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/rolling_outputs/WalkForward_All_Strategies_drawdown.png)
![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/rolling_outputs/WalkForward_All_Strategies_sharpe.png)

As we can see, the cumulative returns of each z-score band region is what we expect to see, the pnl is the largest when the z-score bands are the most flexible. As a result, we use a z-entry of 1.5 and z-exit of 0.5 throughout our analysis as it provides the best of both worlds, in terms of risk and return (which we found in our sensitivity analysis in the backtest notebook):

![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/z-sensitivity.png)

# Running project on your own device

Start by creating a virtual environment (venv)

```
py -m venv venv
venv source/bin/activate

pip install -r requirements.txt
```

Before running, ensure you create a `local_config.py` file that directs the data to where the cloned repository is stored:

```
from pathlib import Path
PROJECT_ROOT = Path({your_path_here})
```

To run, you can either:
a) Run through the notebooks individually or,
b) Run through the notebooks using the **run_pipeline.py** script:

```
py run_pipeline.py
```

*Note to run these, you must have the data used, which has been excluded from the repository due to licensing reasons*

To pull the correct data, find the following indexes on Bloomberg Terminal (*which were pulled into excel to export as a .csv*) to replicate pipeline:
*Excel Function*

```
=BDH(B$1,"PX_LAST","01/01/2015","04/20/2026","CapChg=Y","CshAdjNormal=Y","CshAdjAbnormal=Y")
```

| Sector    | Index Universe                | Bloomberg Screen |
| --------- | ----------------------------- | ---------------- |
| Tech      | Hang-Seng Tech Index          | HSTECH Index     |
| Commodity | Hang Seng Composite Energy    | HSCIE Index      |
| Materials | Hang-Seng Composite Materials | HSCIM Index      |

The following show the entire investment universe used (pulling the top 30 by market cap from commodity and materials combined):

<details>
<summary>Toggle Here</summary>

Full Investment Universe:

| Ticker (Tech)  | Company (Tech)                  | Ticker (Energy/Materials) | Company (Energy/Materials)   |
| -------------- | ------------------------------- | ------------------------- | ---------------------------- |
| 700 HK Equity  | TENCENT HOLDINGS LTD            | 857 HK Equity             | PETROCHINA CO LTD-H          |
| 9988 HK Equity | ALIBABA GROUP HOLDING LTD       | 883 HK Equity             | CNOOC LTD-H                  |
| 1211 HK Equity | BYD CO LTD-H                    | 1088 HK Equity            | CHINA SHENHUA ENERGY CO-H    |
| 1810 HK Equity | XIAOMI CORP-CLASS B             | 2899 HK Equity            | ZIJIN MINING GROUP CO LTD-H  |
| 300 HK Equity  | MIDEA GROUP CO LTD-H            | 386 HK Equity             | CHINA PETROLEUM & CHEMICAL-H |
| 981 HK Equity  | SEMICONDUCTOR MANUFACTURING-H   | 3993 HK Equity            | CMOC GROUP LTD-H             |
| 9999 HK Equity | NETEASE INC                     | 2259 HK Equity            | ZIJIN GOLD INTERNATIONAL CO  |
| 3690 HK Equity | MEITUAN-CLASS B                 | 1378 HK Equity            | CHINA HONGQIAO GROUP LTD     |
| 9618 HK Equity | JD.COM INC-CLASS A              | 2600 HK Equity            | ALUMINUM CORP OF CHINA LTD-H |
| 9888 HK Equity | BAIDU INC-CLASS A               | 1898 HK Equity            | CHINA COAL ENERGY CO-H       |
| 9961 HK Equity | TRIP.COM GROUP LTD              | 1772 HK Equity            | GANFENG LITHIUM GROUP CO L-H |
| 6690 HK Equity | HAIER SMART HOME CO LTD-H       | 1787 HK Equity            | SHANDONG GOLD MINING CO LT-H |
| 1024 HK Equity | KUAISHOU TECHNOLOGY             | 1171 HK Equity            | YANKUANG ENERGY GROUP CO-H   |
| 1347 HK Equity | HUA HONG SEMICONDUCTOR LTD-H    | 358 HK Equity             | JIANGXI COPPER CO LTD-H      |
| 2015 HK Equity | LI AUTO INC-CLASS A             | 1818 HK Equity            | ZHAOJIN MINING INDUSTRY CO-H |
| 6618 HK Equity | JD HEALTH INTERNATIONAL INC     | 1208 HK Equity            | MMG LTD                      |
| 992 HK Equity  | LENOVO GROUP LTD                | 2099 HK Equity            | CHINA GOLD INTERNATIONAL RES |
| 9866 HK Equity | NIO INC-CLASS A                 | 2883 HK Equity            | CHINA OILFIELD SERVICES-H    |
| 9868 HK Equity | XPENG INC - CLASS A SHARES      | 2788 HK Equity            | CHUANGXIN INDUSTRIES HOLDING |
| 1698 HK Equity | TENCENT MUSIC ENT - CLASS A     | 3939 HK Equity            | WANGUO GOLD GROUP LTD        |
| 9660 HK Equity | HORIZON ROBOTICS INC            | 3858 HK Equity            | JIAXIN INTERNATIONAL RESOURC |
| 20 HK Equity   | SENSE TIME GROUP INC-CLASS B    | 1258 HK Equity            | CHINA NONFERROUS MINING CORP |
| 9626 HK Equity | BILIBILI INC-CLASS Z            | 3668 HK Equity            | YANCOAL AUSTRALIA LTD        |
| 241 HK Equity  | ALIBABA HEALTH INFORMATION TECH | 3330 HK Equity            | LINGBAO GOLD GROUP CO LTD-H  |
| 9863 HK Equity | ZHEJIANG LEAPMOTOR TECHNOLOGY-H | 1164 HK Equity            | CGN MINING CO LTD            |
| 2382 HK Equity | SUNNY OPTICAL TECH              | 2689 HK Equity            | NINE DRAGONS PAPER HOLDINGS  |
| 285 HK Equity  | BYD ELECTRONIC INTL CO LTD      | 2386 HK Equity            | SINOPEC ENGINEERING GROUP-H  |
| 780 HK Equity  | TONGCHENG TRAVEL HOLDINGS LTD   | 2610 HK Equity            | NANSHAN ALUMINIUM INTERNATIO |
| 3888 HK Equity | KINGSOFT CORP LTD               | 3899 HK Equity            | CIMC ENRIC HOLDINGS LTD      |
| 268 HK Equity  | KINGDEE INTERNATIONAL SOFTWARE  | 189 HK Equity             | DONGYUE GROUP                |

</details>

# Outline of Pipeline

![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/pipeline.png)
