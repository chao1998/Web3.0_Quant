import numpy as np
import pandas as pd
import datetime
import os
import time
import multiprocessing as mp
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import shap
import warnings
warnings.filterwarnings('ignore')

# Fix matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class OptimizedModel:
    def __init__(self):
        self.train_data_path = "/tmp/train_data"
        self.submission_id_path = "./submission_id.csv"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.scaler = RobustScaler()
        
        # Disable PyTorch for now to avoid issues
        self.use_torch = False
        self.device = 'cpu'
        print("Using NumPy for computation")
    
    def get_all_symbol_list(self):
        try:
            parquet_name_list = os.listdir(self.train_data_path)
            symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list if parquet_name.endswith('.parquet')]
            return symbol_list
        except Exception as e:
            print(f"Error in get_all_symbol_list: {e}")
            return []

    def compute_factors_numpy(self, df):
        """Compute technical indicators using NumPy with fixed pandas compatibility"""
        try:
            # Extract data as pandas Series first, then convert to numpy for calculations
            close_series = df['close_price'].copy()
            volume_series = df['volume'].copy()
            amount_series = df['amount'].copy()
            high_series = df['high_price'].copy()
            low_series = df['low_price'].copy()
            buy_volume_series = df['buy_volume'].copy()
            
            # Check for valid data
            if len(close_series) < 50:
                raise ValueError("Insufficient data points")
            
            # Fill missing values using pandas methods
            close_series = close_series.fillna(method='ffill').fillna(method='bfill')
            volume_series = volume_series.fillna(0).clip(lower=0)
            amount_series = amount_series.fillna(0).clip(lower=0)
            high_series = high_series.fillna(method='ffill').fillna(close_series)
            low_series = low_series.fillna(method='ffill').fillna(close_series)
            buy_volume_series = buy_volume_series.fillna(volume_series/2).clip(lower=0, upper=volume_series)
            
            # Ensure price consistency using pandas operations
            high_series = pd.Series(np.maximum(high_series.values, close_series.values), index=high_series.index)
            low_series = pd.Series(np.minimum(low_series.values, close_series.values), index=low_series.index)
            
            # Convert to numpy arrays for calculations
            close = close_series.values
            volume = volume_series.values
            amount = amount_series.values
            high = high_series.values
            low = low_series.values
            buy_volume = buy_volume_series.values
            
            if np.all(close == 0) or np.all(np.isnan(close)):
                raise ValueError("Invalid close price data")
            
            # VWAP calculation
            vwap = np.where(volume > 0, amount / volume, close)
            vwap = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close)
            
            # RSI calculation using pandas for rolling operations
            def calculate_rsi(price_series, period=14):
                delta = price_series.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # Use pandas rolling operations
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                return rsi.fillna(50).values
            
            rsi = calculate_rsi(close_series)
            
            # MACD calculation using pandas EWM
            ema_12 = close_series.ewm(span=12, adjust=False).mean()
            ema_26 = close_series.ewm(span=26, adjust=False).mean()
            macd = (ema_12 - ema_26).values
            
            # ATR calculation
            def calculate_atr(high_series, low_series, close_series, period=14):
                prev_close = close_series.shift(1)
                
                tr1 = high_series - low_series
                tr2 = (high_series - prev_close).abs()
                tr3 = (low_series - prev_close).abs()
                
                # Use pandas operations for max
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(window=period, min_periods=1).mean()
                return atr.values
            
            atr = calculate_atr(high_series, low_series, close_series)
            
            # Bollinger Bands using pandas rolling
            def calculate_bollinger_bands(price_series, period=20):
                sma = price_series.rolling(window=period, min_periods=1).mean()
                std = price_series.rolling(window=period, min_periods=1).std()
                bb_width = (std / (sma + 1e-8)).fillna(0)
                return bb_width.values
            
            bb_width = calculate_bollinger_bands(close_series)
            
            # Volume indicators using pandas rolling
            volume_sma = volume_series.rolling(window=20, min_periods=1).mean()
            volume_ratio = (volume_series / (volume_sma + 1e-8)).fillna(1).values
            
            # Buy pressure indicators
            buy_ratio = np.where(volume > 0, np.clip(buy_volume / volume, 0, 1), 0.5)
            buy_pressure = (buy_volume - (volume - buy_volume)) / (volume + 1e-8)
            
            # Price position indicators
            price_position = np.where((high - low) > 0, (close - low) / (high - low), 0.5)
            
            # VWAP deviation
            vwap_deviation = (close - vwap) / (vwap + 1e-8)
            
            # Momentum indicators using pandas shift
            momentum_5 = ((close_series / close_series.shift(5)) - 1).fillna(0).values
            momentum_20 = ((close_series / close_series.shift(20)) - 1).fillna(0).values
            
            # Returns calculation
            returns = close_series.pct_change().fillna(0).values
            
            # Assign computed factors to DataFrame
            factor_dict = {
                'vwap': vwap,
                'rsi': rsi,
                'macd': macd,
                'atr': atr,
                'bb_width': bb_width,
                'volume_ratio': volume_ratio,
                'buy_ratio': buy_ratio,
                'buy_pressure': buy_pressure,
                'price_position': price_position,
                'vwap_deviation': vwap_deviation,
                'momentum_5': momentum_5,
                'momentum_20': momentum_20,
                'returns': returns
            }
            
            # Validate and clean all factors
            for name, factor in factor_dict.items():
                # Replace inf and nan
                factor = np.where(np.isfinite(factor), factor, 0)
                
                # Clip extreme values (except for bounded indicators)
                if name not in ['rsi', 'buy_ratio', 'price_position']:
                    if np.any(factor != 0):
                        q1, q99 = np.percentile(factor[factor != 0], [1, 99])
                        if q99 > q1:
                            factor = np.clip(factor, q1, q99)
                
                df[name] = factor
            
            return df
            
        except Exception as e:
            print(f"Error in compute_factors_numpy: {e}")
            # Return DataFrame with safe default values
            n_rows = len(df)
            default_factors = {
                'vwap': df['close_price'].fillna(method='ffill').fillna(0),
                'rsi': pd.Series(np.full(n_rows, 50.0), index=df.index),
                'macd': pd.Series(np.zeros(n_rows), index=df.index),
                'atr': pd.Series(np.full(n_rows, 0.01), index=df.index),
                'bb_width': pd.Series(np.full(n_rows, 0.1), index=df.index),
                'volume_ratio': pd.Series(np.ones(n_rows), index=df.index),
                'buy_ratio': pd.Series(np.full(n_rows, 0.5), index=df.index),
                'buy_pressure': pd.Series(np.zeros(n_rows), index=df.index),
                'price_position': pd.Series(np.full(n_rows, 0.5), index=df.index),
                'vwap_deviation': pd.Series(np.zeros(n_rows), index=df.index),
                'momentum_5': pd.Series(np.zeros(n_rows), index=df.index),
                'momentum_20': pd.Series(np.zeros(n_rows), index=df.index),
                'returns': pd.Series(np.zeros(n_rows), index=df.index)
            }
            
            for name, factor in default_factors.items():
                df[name] = factor
            
            return df

    def get_single_symbol_kline_data(self, symbol):
        try:
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = df.astype(np.float64)
            
            required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount', 'buy_volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return pd.DataFrame()
            
            # Enhanced data quality checks
            if len(df) < 200:
                return pd.DataFrame()
                
            # Check for valid price data
            valid_prices = df['close_price'].dropna()
            if len(valid_prices) < len(df) * 0.7:
                return pd.DataFrame()
                
            if (valid_prices <= 0).sum() > len(valid_prices) * 0.05:
                return pd.DataFrame()
            
            # Fill missing values before outlier detection
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # More conservative outlier removal using pandas operations
            for col in ['close_price', 'high_price', 'low_price', 'open_price']:
                if col in df.columns:
                    q01 = df[col].quantile(0.005)
                    q99 = df[col].quantile(0.995)
                    if q99 > q01:
                        df[col] = df[col].clip(q01, q99)
            
            # Volume outlier handling using pandas operations
            for col in ['volume', 'amount', 'buy_volume']:
                if col in df.columns:
                    # Use log transformation for volume data
                    log_col = np.log1p(df[col])
                    q01 = log_col.quantile(0.01)
                    q99 = log_col.quantile(0.99)
                    log_col_clipped = log_col.clip(q01, q99)
                    df[col] = np.expm1(log_col_clipped)
            
            # Ensure price consistency using pandas operations
            df['high_price'] = df[['high_price', 'close_price']].max(axis=1)
            df['low_price'] = df[['low_price', 'close_price']].min(axis=1)
            
            # Compute factors
            df = self.compute_factors_numpy(df)
            
            # Final validation
            required_factors = ['vwap', 'rsi', 'macd', 'atr', 'buy_ratio', 'returns']
            for factor in required_factors:
                if factor not in df.columns:
                    return pd.DataFrame()
            
            # Check for reasonable factor values
            if df['vwap'].isna().all() or (df['vwap'] <= 0).all():
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return pd.DataFrame()

    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
        
        all_symbol_list = self.get_all_symbol_list()
        if not all_symbol_list:
            print("No symbols found, exiting.")
            return [], [], [], [], [], [], [], []
        
        print(f"Loading {len(all_symbol_list)} symbols...")
        
        # Load data sequentially for better debugging
        valid_symbols = []
        valid_dfs = []
        
        for i, symbol in enumerate(all_symbol_list):
            if i % 50 == 0:
                print(f"Processing symbol {i+1}/{len(all_symbol_list)}: {symbol}")
            
            df = self.get_single_symbol_kline_data(symbol)
            if not df.empty:
                valid_symbols.append(symbol)
                valid_dfs.append(df)
        
        print(f"Successfully loaded {len(valid_symbols)} out of {len(all_symbol_list)} symbols")
        
        if len(valid_symbols) == 0:
            print("No valid symbols loaded!")
            return [], [], [], [], [], [], [], []

        # Create time index
        time_index = pd.date_range(start=self.start_datetime, end='2024-12-31', freq='15min')
        
        def create_aligned_array(column_name):
            try:
                aligned_data = []
                
                for symbol, df in zip(valid_symbols, valid_dfs):
                    if column_name in df.columns:
                        # Reindex to time_index with forward fill
                        series = df[column_name].reindex(time_index, method='ffill')
                        # Fill remaining NaN with appropriate defaults
                        if column_name == 'rsi':
                            series = series.fillna(50)
                        elif column_name in ['buy_ratio', 'price_position']:
                            series = series.fillna(0.5)
                        elif column_name == 'volume_ratio':
                            series = series.fillna(1.0)
                        else:
                            series = series.fillna(0)
                        aligned_data.append(series)
                
                if not aligned_data:
                    return np.zeros((len(time_index), len(all_symbol_list)))
                
                # Combine data
                combined_df = pd.concat(aligned_data, axis=1)
                combined_df.columns = valid_symbols
                
                # Reindex to include all symbols
                combined_df = combined_df.reindex(columns=all_symbol_list, fill_value=0)
                
                return combined_df.values
                
            except Exception as e:
                print(f"Error creating array for {column_name}: {e}")
                return np.zeros((len(time_index), len(all_symbol_list)))

        print("Creating aligned arrays...")
        
        # Create arrays for all features
        arrays = {}
        feature_names = ['vwap', 'amount', 'volume', 'buy_volume', 'rsi', 'macd', 'atr', 
                        'buy_ratio', 'buy_pressure', 'returns', 'momentum_5', 'momentum_20']
        
        for feature in feature_names:
            arrays[feature] = create_aligned_array(feature)
            non_zero = np.count_nonzero(arrays[feature])
            print(f"{feature}: shape={arrays[feature].shape}, non-zero={non_zero}")
        
        time_arr = pd.to_datetime(time_index).values
        
        print(f"Data loading completed in {datetime.datetime.now() - t0}")
        
        # Return the main arrays needed
        return (all_symbol_list, time_arr, arrays['vwap'], arrays['amount'], 
                arrays['atr'], arrays['macd'], arrays['buy_volume'], arrays['volume'],
                arrays)

    def create_advanced_features(self, df_dict, time_arr, all_symbol_list):
        """Create more sophisticated features with fixed iteration"""
        print("Creating advanced features...")
        
        # Create DataFrames from arrays
        features = {}
        
        # Basic features
        df_vwap = pd.DataFrame(df_dict['vwap'], columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(df_dict['amount'], columns=all_symbol_list, index=time_arr)
        df_volume = pd.DataFrame(df_dict['volume'], columns=all_symbol_list, index=time_arr)
        df_returns = pd.DataFrame(df_dict['returns'], columns=all_symbol_list, index=time_arr)
        
        # Time windows (in 15-minute intervals)
        windows = {
            '1h': 4,      # 4 * 15min
            '4h': 16,     # 16 * 15min
            '1d': 96,     # 96 * 15min
            '3d': 288,    # 288 * 15min
            '7d': 672     # 672 * 15min
        }
        
        # Multi-timeframe momentum using pandas operations
        for name, window in windows.items():
            if window < len(df_vwap):
                momentum = (df_vwap / df_vwap.shift(window) - 1).fillna(0)
                features[f'momentum_{name}'] = momentum
                non_zero = np.count_nonzero(momentum.values)
                print(f"momentum_{name}: non-zero={non_zero}")
        
        # Volatility features using pandas rolling
        for name, window in windows.items():
            if window < len(df_returns):
                volatility = df_returns.rolling(window, min_periods=1).std().fillna(0)
                features[f'volatility_{name}'] = volatility
        
        # Volume features using pandas operations
        vol_sma_1d = df_volume.rolling(windows['1d'], min_periods=1).mean()
        vol_ratio = (df_volume / (vol_sma_1d + 1e-8)).fillna(1)
        features['volume_anomaly'] = vol_ratio
        
        # Cross-sectional ranking features
        # FIX: Create a copy of features dict to avoid iteration modification
        momentum_features = {k: v for k, v in features.items() if 'momentum' in k}
        
        for name, momentum_df in momentum_features.items():
            # Rank within each timestamp using pandas
            ranks = momentum_df.rank(axis=1, pct=True).fillna(0.5)
            features[f'{name}_rank'] = ranks
        
        # Price-based features
        features['price_momentum_short'] = (df_vwap / df_vwap.shift(16) - 1).fillna(0)  # 4h
        features['price_momentum_long'] = (df_vwap / df_vwap.shift(672) - 1).fillna(0)  # 7d
        
        # Volume-based features
        features['volume_momentum'] = (df_volume / df_volume.shift(96) - 1).fillna(0)  # 1d
        features['amount_momentum'] = (df_amount / df_amount.shift(96) - 1).fillna(0)  # 1d
        
        # Moving averages
        features['vwap_ma_short'] = df_vwap.rolling(windows['4h'], min_periods=1).mean()
        features['vwap_ma_long'] = df_vwap.rolling(windows['1d'], min_periods=1).mean()
        
        # Relative strength features
        features['relative_strength'] = (features['vwap_ma_short'] / features['vwap_ma_long'] - 1).fillna(0)
        
        print(f"Created {len(features)} advanced features")
        
        return features

    def weighted_spearmanr(self, y_true, y_pred):
        """Calculate weighted Spearman correlation"""
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            if len(y_true) < 2:
                return 0
            
            n = len(y_true)
            r_true = pd.Series(y_true).rank(ascending=False, method='average')
            r_pred = pd.Series(y_pred).rank(ascending=False, method='average')
            
            x = 2 * (r_true - 1) / (n - 1) - 1
            w = x ** 2
            w_sum = w.sum()
            
            if w_sum == 0:
                return 0
            
            mu_true = (w * r_true).sum() / w_sum
            mu_pred = (w * r_pred).sum() / w_sum
            cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
            var_true = (w * (r_true - mu_true)**2).sum()
            var_pred = (w * (r_pred - mu_pred)**2).sum()
            
            if var_true * var_pred <= 0:
                return 0
            
            return cov / np.sqrt(var_true * var_pred)
        except Exception as e:
            print(f"Error in weighted_spearmanr: {e}")
            return 0

    def analyze_submission_ids(self):
        """Analyze submission ID format to understand the mismatch"""
        if not os.path.exists(self.submission_id_path):
            print(f"Submission ID file not found: {self.submission_id_path}")
            return None
            
        df_submission_id = pd.read_csv(self.submission_id_path)
        print(f"Submission ID file analysis:")
        print(f"  Total IDs: {len(df_submission_id)}")
        print(f"  Sample IDs:")
        
        sample_ids = df_submission_id['id'].head(10).tolist()
        for i, id_str in enumerate(sample_ids):
            print(f"    {i+1}: {id_str}")
            
        # Analyze ID format
        if len(sample_ids) > 0:
            first_id = sample_ids[0]
            if '_' in first_id:
                parts = first_id.split('_')
                print(f"  ID format appears to be: {parts[0]}_{parts[1] if len(parts) > 1 else '?'}")
                print(f"  Datetime part: {parts[0]}")
                print(f"  Symbol part: {parts[1] if len(parts) > 1 else 'Not found'}")
                
                # Try to parse datetime
                try:
                    datetime_str = parts[0]
                    if len(datetime_str) == 14:  # YYYYMMDDHHMMSS
                        parsed_dt = pd.to_datetime(datetime_str, format='%Y%m%d%H%M%S')
                        print(f"  Parsed datetime: {parsed_dt}")
                except:
                    print(f"  Could not parse datetime from: {datetime_str}")
        
        return df_submission_id

    def train(self, target_features, all_features):
        print("Preparing training data...")
        
        # Combine all features
        feature_dfs = []
        feature_names = []
        
        for name, df in all_features.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                stacked = df.stack()
                stacked.name = name
                feature_dfs.append(stacked)
                feature_names.append(name)
                
                non_zero = np.count_nonzero(df.values)
        
        # Stack target
        target_stacked = target_features.stack()
        target_stacked.name = 'target'
        
        # Combine all data
        all_data = feature_dfs + [target_stacked]
        data = pd.concat(all_data, axis=1)
        
        print(f"Combined data shape: {data.shape}")
        
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN
        original_size = len(data)
        data = data.dropna()
        print(f"Data size: {original_size} -> {len(data)} (dropped {original_size - len(data)})")
        
        if len(data) == 0:
            print("No valid data after cleaning!")
            return
        
        # Prepare features and target
        X = data[feature_names]
        y = data['target']
        
        # Check target distribution
        print(f"Target distribution:")
        print(f"  Mean: {y.mean():.6f}")
        print(f"  Std: {y.std():.6f}")
        print(f"  Min: {y.min():.6f}")
        print(f"  Max: {y.max():.6f}")
        print(f"  Non-zero: {np.count_nonzero(y)}/{len(y)}")
        
        if y.std() < 1e-8:
            print("WARNING: Target has very low variance!")
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-validation split (temporal)
        split_point = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
        
        print(f"Training size: {len(X_train)}, Validation size: {len(X_val)}")
        
        # Create sample weights (emphasize extreme returns)
        y_abs = np.abs(y_train)
        threshold = y_abs.quantile(0.8)
        sample_weights = np.where(y_abs > threshold, 3.0, 1.0)
        
        # Train model with updated XGBoost API
        print("Training model...")
        try:
            # Try new XGBoost API first
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                early_stopping_rounds=20,  # Try this first
                eval_metric='rmse'
            )
            
            model.fit(
                X_train, y_train, 
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)], 
                verbose=50
            )
            
        except TypeError:
            # Fallback to older API
            print("Using older XGBoost API...")
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                reg_alpha=0.1,
                reg_lambda=1.0
            )
            
            model.fit(
                X_train, y_train, 
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)], 
                verbose=50
            )
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_all = model.predict(X_scaled)
        
        print(f"\nPrediction statistics:")
        print(f"Train predictions: min={y_pred_train.min():.6f}, max={y_pred_train.max():.6f}, "
              f"mean={y_pred_train.mean():.6f}, std={y_pred_train.std():.6f}")
        print(f"Val predictions: min={y_pred_val.min():.6f}, max={y_pred_val.max():.6f}, "
              f"mean={y_pred_val.mean():.6f}, std={y_pred_val.std():.6f}")
        print(f"Non-zero predictions: train={np.count_nonzero(y_pred_train)}, val={np.count_nonzero(y_pred_val)}")
        
        # Calculate correlations
        train_corr = self.weighted_spearmanr(y_train, y_pred_train)
        val_corr = self.weighted_spearmanr(y_val, y_pred_val)
        print(f"Correlations: train={train_corr:.4f}, validation={val_corr:.4f}")
        
        # Feature importance
        try:
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 feature importances:")
            print(feature_importance.head(10))
        except:
            print("Could not get feature importance")
        
        # Add predictions to data
        data['y_pred'] = y_pred_all
        
        # Analyze submission IDs before processing
        print("\n" + "="*50)
        print("ANALYZING SUBMISSION ID FORMAT")
        print("="*50)
        df_submission_id = self.analyze_submission_ids()
        
        # Prepare submission
        print("Preparing submission...")
        df_submit = data.reset_index()
        df_submit = df_submit.rename(columns={'level_0': 'datetime', 'level_1': 'symbol'})
        df_submit = df_submit[['datetime', 'symbol', 'y_pred']].copy()
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        
        # Filter by date
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        
        # Create submission ID with debugging
        print(f"Creating submission IDs...")
        print(f"Sample datetime: {df_submit['datetime'].iloc[0]}")
        print(f"Sample symbol: {df_submit['symbol'].iloc[0]}")
        
        # Create ID in the same format as the submission file
        df_submit['id'] = df_submit['datetime'].dt.strftime('%Y%m%d%H%M%S') + '_' + df_submit['symbol']
        
        # Show sample IDs
        print(f"Sample generated IDs:")
        for i, id_str in enumerate(df_submit['id'].head(5)):
            print(f"  {i+1}: {id_str}")
        
        df_submit = df_submit[['id', 'predict_return']]
        
        print(f"Submission before ID filtering: {len(df_submit)} rows")
        print(f"Prediction range: [{df_submit['predict_return'].min():.6f}, {df_submit['predict_return'].max():.6f}]")
        print(f"Non-zero predictions: {np.count_nonzero(df_submit['predict_return'])}")
        
        # Handle submission IDs with detailed debugging
        if df_submission_id is not None:
            required_ids = set(df_submission_id['id'].tolist())
            available_ids = set(df_submit['id'].tolist())
            
            print(f"Required IDs: {len(required_ids)}")
            print(f"Available IDs: {len(available_ids)}")
            overlap_ids = required_ids & available_ids
            print(f"Overlap: {len(overlap_ids)}")
            
            if len(overlap_ids) == 0:
                print("WARNING: NO ID OVERLAP! Checking for format issues...")
                
                # Sample some IDs to check format differences
                sample_required = list(required_ids)[:5]
                sample_available = list(available_ids)[:5]
                
                print("Sample required IDs:")
                for id_str in sample_required:
                    print(f"  {id_str}")
                
                print("Sample available IDs:")
                for id_str in sample_available:
                    print(f"  {id_str}")
                
                # Try to find the issue
                if len(sample_required) > 0 and len(sample_available) > 0:
                    req_parts = sample_required[0].split('_')
                    avail_parts = sample_available[0].split('_')
                    
                    print(f"Required ID parts: {req_parts}")
                    print(f"Available ID parts: {avail_parts}")
                    
                    if len(req_parts) > 1 and len(avail_parts) > 1:
                        print(f"Datetime comparison: {req_parts[0]} vs {avail_parts[0]}")
                        print(f"Symbol comparison: {req_parts[1]} vs {avail_parts[1]}")
            
            # Filter to required IDs
            df_submit_final = df_submit[df_submit['id'].isin(required_ids)].copy()
            
            # Add missing IDs with zero predictions
            missing_ids = required_ids - available_ids
            if missing_ids:
                print(f"Adding {len(missing_ids)} missing IDs with zero predictions")
                missing_df = pd.DataFrame({
                    'id': list(missing_ids),
                    'predict_return': 0.0
                })
                df_submit_final = pd.concat([df_submit_final, missing_df], ignore_index=True)
        else:
            print("No submission ID file found, using all predictions")
            df_submit_final = df_submit
        
        print(f"\nFinal submission statistics:")
        print(f"Total rows: {len(df_submit_final)}")
        print(f"Prediction range: [{df_submit_final['predict_return'].min():.6f}, {df_submit_final['predict_return'].max():.6f}]")
        print(f"Non-zero predictions: {np.count_nonzero(df_submit_final['predict_return'])}")
        print(f"Mean prediction: {df_submit_final['predict_return'].mean():.6f}")
        print(f"Std prediction: {df_submit_final['predict_return'].std():.6f}")
        
        # Save submission
        df_submit_final.to_csv('final_submission_R8KQUF.csv', index=False)
        print("Submission saved to submit.csv")
        
        # Show a few sample rows from final submission
        print("\nSample submission rows:")
        print(df_submit_final.head(10))
        
        # Overall correlation
        overall_corr = self.weighted_spearmanr(data['target'], data['y_pred'])
        print(f"Overall correlation: {overall_corr:.4f}")

    def run(self):
        # Load data
        all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr, all_arrays = self.get_all_symbol_kline()
        
        if not all_symbol_list:
            print("No data loaded, exiting.")
            return
        
        print(f"Loaded {len(all_symbol_list)} symbols")
        
        # Create advanced features
        advanced_features = self.create_advanced_features(all_arrays, time_arr, all_symbol_list)
        
        # Calculate target (future returns)
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        target_periods = 96  # 1 day ahead (96 * 15min)
        
        # Use 1-day forward return as target
        future_returns = (df_vwap.shift(-target_periods) / df_vwap - 1).fillna(0)
        
        print(f"Target statistics:")
        print(f"  Mean: {future_returns.mean().mean():.6f}")
        print(f"  Std: {future_returns.std().mean():.6f}")
        print(f"  Non-zero: {np.count_nonzero(future_returns.values)}")
        
        # Train model
        self.train(future_returns, advanced_features)


if __name__ == '__main__':
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    train_data_path = "/tmp/train_data"
    if not os.path.exists(train_data_path):
        print(f"Train data path does not exist: {train_data_path}")
        exit(1)
    
    print("Starting Enhanced Crypto Prediction Model...")
    model = OptimizedModel()
    model.run()
    print("Model execution completed!")