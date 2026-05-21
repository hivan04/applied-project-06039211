# Statistical Arbitrage using Co-Integration in HK Tech & Commodities

The main question of this project is to see whether statistical arbitrage works in a highly volatile and emerging sector environement such as Hong Kong Technology and Commodities.

# Contents
- [[Key Results from Backtesting]]
- [[Limitations of Project]]

# Key Results from Backtesting 
In this project, 3 backtests were ran (*all three backtests were run with R = 1, where R is the observation noise covariance in the Kalman Filter. This value was chosen as a neutral baseline, reflecting equal confidence in observed price updates and the model's prior estimates.*):

1. Baseline Backtest - this encompasses a static z-score entry/exit and trades based off primarily the Kalman Filter 
2. Refined Backtest - this builds off the baseline strategy by including different layers of risk management to the signal
   a) The first refinemnent to the trading signal was implementing a dyanmic z-score band, the same as the baseline but the threshold for entry/exit adjusts accordingly depending on the time-series (of course with no look-ahead bias);
   b) The second refinement was calculating the dynamic maximum drawdown (via. estimated volatility)
3. Rolling Window Backtest (12-Month Window) - backtested both the baseline and refined trading signals using rolling window analysis.


## Baseline Backtest 
![image]()

## Refined Backtest
![image]()

## Rolling-Window Backtest 
![image]()

# Limitations of Project :
*write in conclusion*
- Lack of regime analysis
- Enhanced sensitivity analyis
- Potentially question the Kalman filter and if there are any other filters that are more viable for our investment universe
- Investment Universe limits our reasoning and statistical evidence for broader asset classes
- NLP LLM Judge for Economic Reasoning (via. Prompt Engineering)

![image](https://github.com/hivan04/applied-project-06039211/blob/main/outputs/proj-outline.png)
