# Strategy Overview

This script implements an enhanced cryptocurrency return prediction strategy using 
multi-timeframe technical indicators, advanced feature engineering, and an XGBoost 
regression model. It processes high-frequency (15-minute) k-line data for multiple 
symbols, computes a broad set of factors, and predicts future returns over a 1-day horizon.

---------------------------------------------------------------------------------------------

## 1. Data Loading & Preprocessing

- Reads `.parquet` k-line files for all available symbols.
- Cleans and validates data:
  - Removes extreme outliers via quantile clipping.
  - Forward/backward fills missing prices.
  - Ensures price consistency (`high_price ≥ close_price ≥ low_price`).
  - Handles skewed volume data via log transform clipping.
  - Aligns all symbols to a common 15-minute time index for consistent feature generation.

----------------------------------------------------------------------------------------------

## 2. Base Factor Computation

For each symbol, the following indicators are computed:

- VWAP (Volume-Weighted Average Price) and VWAP deviation.
- RSI (Relative Strength Index, 14-period).
- MACD (12–26 EMA difference).
- ATR (Average True Range, 14-period).
- Bollinger Band width (20-period).
- Volume ratio (volume / 20-period average).
- Buy pressure metrics (buy ratio, buy volume difference).
- Price position within daily range.
- Momentum (5-period & 20-period returns).
- Simple returns (period-to-period).

-----------------------------------------------------------------------------------------------

## 3. Advanced Feature Engineering

From aligned arrays, the model builds multi-horizon features:

- Multi-timeframe momentum: 1h, 4h, 1d, 3d, 7d.
- Rolling volatility for each horizon.
- Volume anomaly ratio (current volume vs. 1-day SMA).
- Cross-sectional momentum ranks (relative to all symbols at each timestamp).
- Price momentum (short-term: 4h; long-term: 7d).
- Volume & amount momentum (1-day).
- VWAP moving averages (short: 4h; long: 1d) and relative strength between them.

------------------------------------------------------------------------------------------------

## 4. Target Definition

Prediction target:

```
1-day forward VWAP return = (VWAP_t+96 − VWAP_t) / VWAP_t
```
where `96` = number of 15-minute intervals in a day.

--------------------------------------------------------------------------------------------------

## 5. Model Training

- Combines all engineered features into a single dataset.
- Cleans NaNs/Infs, scales features using `RobustScaler`.
- Splits temporally (80% train / 20% validation).
- Applies sample weighting to emphasize top 20% absolute returns.
- Trains XGBoost Regressor:
  - `max_depth=6`
  - `learning_rate=0.1`
  - `n_estimators=200`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - L1/L2 regularization
- Uses RMSE for evaluation and early stopping.

-----------------------------------------------------------------------------------------------------

## 6. Evaluation

- Calculates weighted Spearman correlation between predictions and actual returns (weights emphasize high-magnitude returns).
- Logs:
  - Train and validation correlations
  - Prediction distributions
  - Top 10 feature importances

------------------------------------------------------------------------------------------------------

## 7. Submission Generation

- Creates predictions for all aligned timestamps and symbols.
- Matches output IDs to a provided `submission_id.csv` file.
- Adds missing IDs with zero predictions if needed.
- Saves the final CSV with `id` and `predict_return` columns.

------------------------------------------------------------------------------------------------------

## Key Characteristics of the Strategy

- **Multi-timeframe design**: Uses short, medium, and long horizons to capture both short-term momentum shifts and long-term trends.
- **Cross-sectional ranking**: Incorporates relative performance between assets.
- **Volatility & anomaly detection**: Captures unusual market activity.
- **Weighted correlation metric**: Optimizes for high-impact predictions rather than average accuracy.
- **Robust cleaning**: Aggressive outlier handling to avoid distorted signals.


Author By Following Team Member:
1、Chao Wei, R8KQUF, 19970415762@163.com, Monash University Malaysia, Chinese.
2、Boxuan Li, L90170, 2639556380@qq.com, Xidian University, Chinese.
3、Guohao Qi, M4WVL, qgh124430@hnu.edu.cn, National University of Singapore, Chinese.