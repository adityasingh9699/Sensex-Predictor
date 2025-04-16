import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path
import logging
import asyncio
import pytz
import ta
from app.database.models import HistoricalData, DailyData, NewsData
from app.config.settings import settings
from app.services.data_service import DataService
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.lstm_model = None
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            random_state=42
        )
        self.price_scaler = RobustScaler()
        self.feature_scaler = RobustScaler()
        self.ist_tz = pytz.timezone(settings.TIMEZONE)
        self.load_or_create_model()
        self.training_task = None
        self.min_confidence_threshold = 0.85  # 85% minimum confidence threshold
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            model_path = Path(settings.MODEL_SAVE_PATH)
            model_path.mkdir(parents=True, exist_ok=True)
            model_file = model_path / "model.h5"
            
            create_new = True
            if model_file.exists():
                try:
                    # Try to load the model
                    temp_model = load_model(str(model_file), compile=False)
                    
                    # Check if input shape matches our current feature set
                    expected_shape = (None, settings.SEQUENCE_LENGTH, 18)  # 18 features
                    actual_shape = temp_model.layers[0].input_shape
                    
                    if actual_shape == expected_shape:
                        self.lstm_model = temp_model
                        self.lstm_model.compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss='huber',
                            metrics=['mean_absolute_percentage_error']
                        )
                        self._load_scalers()
                        logger.info("Loaded existing model")
                        create_new = False
                    else:
                        logger.warning(f"Model input shape mismatch. Expected {expected_shape}, got {actual_shape}")
                        # Delete existing model and scalers
                        model_file.unlink(missing_ok=True)
                        (model_path / "price_scaler.pkl").unlink(missing_ok=True)
                        (model_path / "feature_scaler.pkl").unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to load existing model: {str(e)}. Creating new one.")
            
            if create_new:
                self.lstm_model = self._create_model()
                self._save_artifacts()  # Save the new model immediately
                logger.info("Created new model")
                
        except Exception as e:
            logger.error(f"Error in load_or_create_model: {str(e)}")
            raise
    
    def _create_model(self):
        """Create enhanced LSTM model"""
        # Get feature list to ensure correct input shape
        feature_columns = [
            'open', 'high', 'low', 'close',  # Price features
            'volume', 'returns', 'log_returns',  # Volume and returns
            'sma_20', 'sma_50', 'ema_20',  # Trend indicators
            'macd', 'rsi', 'stoch',  # Technical indicators
            'bb_high', 'bb_low', 'atr',  # Volatility indicators
            'price_to_sma20', 'price_to_sma50'  # Price ratios
        ]
        n_features = len(feature_columns)
        
        logger.info(f"Creating model with {n_features} features: {feature_columns}")
        
        # Create a sequential model
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, 
                 input_shape=(settings.SEQUENCE_LENGTH, n_features)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mean_absolute_percentage_error']
        )
        
        # Print model summary for debugging
        model.summary()
        
        return model
    
    def _load_scalers(self):
        """Load saved scalers"""
        scaler_path = Path(settings.MODEL_SAVE_PATH)
        try:
            self.price_scaler = joblib.load(scaler_path / "price_scaler.pkl")
            self.feature_scaler = joblib.load(scaler_path / "feature_scaler.pkl")
        except FileNotFoundError:
            logger.warning("Scalers not found, will create new ones during training")
    
    def prepare_data(self, data: pd.DataFrame, sequence_length: int):
        """Enhanced data preparation with advanced technical indicators"""
        try:
            if data.empty:
                logger.error("Empty DataFrame provided to prepare_data")
                return None, None, None

            # Ensure data is sorted by date
            data = data.sort_index()
            
            # Calculate returns and log returns
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close']).diff()
            
            # Trend Indicators
            data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
            data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
            data['ema_20'] = ta.trend.ema_indicator(data['close'], window=20)
            data['macd'] = ta.trend.macd_diff(data['close'])
            
            # Momentum Indicators
            data['rsi'] = ta.momentum.rsi(data['close'])
            data['stoch'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
            
            # Volatility Indicators
            data['bb_high'] = ta.volatility.bollinger_hband(data['close'])
            data['bb_low'] = ta.volatility.bollinger_lband(data['close'])
            data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
            
            # Price normalization features
            data['price_to_sma20'] = data['close'] / data['sma_20']
            data['price_to_sma50'] = data['close'] / data['sma_50']
            
            # Fill missing values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.ffill().bfill().fillna(0)  # Updated to use newer pandas methods
            
            # Select and order features
            feature_columns = [
                'open', 'high', 'low', 'close',  # Price features
                'volume', 'returns', 'log_returns',  # Volume and returns
                'sma_20', 'sma_50', 'ema_20',  # Trend indicators
                'macd', 'rsi', 'stoch',  # Technical indicators
                'bb_high', 'bb_low', 'atr',  # Volatility indicators
                'price_to_sma20', 'price_to_sma50'  # Price ratios
            ]
            
            # Log feature list for debugging
            logger.info(f"Using features: {feature_columns}")
            
            # Ensure all features exist
            missing_columns = [col for col in feature_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return None, None, None
            
            # Scale features
            try:
                # Use MinMaxScaler for price-related features
                price_features = ['open', 'high', 'low', 'close']
                self.price_scaler.fit(data[price_features])
                
                # Scale all features
                scaled_data = data[feature_columns].copy()
                scaled_data[price_features] = self.price_scaler.transform(data[price_features])
                
                # Scale other features
                other_features = [col for col in feature_columns if col not in price_features]
                self.feature_scaler.fit(data[other_features])
                scaled_data[other_features] = self.feature_scaler.transform(data[other_features])
                
            except Exception as e:
                logger.error(f"Error scaling data: {str(e)}")
                return None, None, None
            
            X, y = [], []
            dates = []
            
            # Create sequences with proper scaling
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data.iloc[i-sequence_length:i][feature_columns].values)
                y.append(scaled_data.iloc[i]['close'])  # Target is scaled close price
                dates.append(data.index[i])
            
            if not X or not y:
                logger.error("No sequences could be created from the data")
                return None, None, None
            
            # Convert to numpy arrays with explicit dtype
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            # Log shapes and sample values for debugging
            logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
            logger.info(f"Number of features: {X.shape[2]}")
            logger.info(f"Sample X values range: [{X.min():.2f}, {X.max():.2f}]")
            logger.info(f"Sample y values range: [{y.min():.2f}, {y.max():.2f}]")
            
            return X, y, self._calculate_weights(dates)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None, None
    
    def _calculate_weights(self, dates):
        """Calculate time-based weights with exponential decay"""
        try:
            current_date = datetime.now().date()
            days_ago = [(current_date - date.date()).days if isinstance(date, datetime)
                       else (current_date - date).days for date in dates]
            
            # Exponential decay with half-life of 30 days
            weights = np.exp(-np.array(days_ago) / 30)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating weights: {str(e)}")
            return np.ones(len(dates)) / len(dates)
    
    async def train_continuously(self):
        """Enhanced continuous training with validation"""
        while self.is_running:
            try:
                now = datetime.now(self.ist_tz)
                
                if now.weekday() < 5:  # Only train on weekdays
                    # Get historical data for training
                    historical_data = self.data_service.get_historical_data_range(
                        start_date=now - timedelta(days=365),
                        end_date=now
                    )
                    
                    if not historical_data.empty:
                        # Prepare data with enhanced features
                        X, y, weights = self.prepare_data(historical_data, settings.SEQUENCE_LENGTH)
                        
                        if X is not None and y is not None:
                            # Create callbacks for better training
                            callbacks = [
                                EarlyStopping(
                                    monitor='val_loss',
                                    patience=5,
                                    restore_best_weights=True
                                ),
                                ReduceLROnPlateau(
                                    monitor='val_loss',
                                    factor=0.5,
                                    patience=3,
                                    min_lr=0.00001
                                )
                            ]
                            
                            # Train model with validation split
                            history = self.lstm_model.fit(
                                X, y,
                                epochs=10,
                                batch_size=32,
                                validation_split=0.2,
                                sample_weight=weights,
                                callbacks=callbacks,
                                verbose=0
                            )
                            
                            # Log training metrics
                            val_loss = history.history['val_loss'][-1]
                            logger.info(f"Model updated with validation loss: {val_loss:.4f}")
                            
                            # Save model if validation loss improved
                            if val_loss < getattr(self, 'best_val_loss', float('inf')):
                                self.best_val_loss = val_loss
                                self._save_artifacts()
                
                await asyncio.sleep(settings.MODEL_UPDATE_INTERVAL * 60)
                
            except Exception as e:
                logger.error(f"Error in continuous training: {str(e)}")
                await asyncio.sleep(60)
    
    def _save_artifacts(self):
        """Save model and scalers"""
        try:
            save_path = Path(settings.MODEL_SAVE_PATH)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.lstm_model.save(save_path / "model.h5")
            
            # Save scalers
            joblib.dump(self.price_scaler, save_path / "price_scaler.pkl")
            joblib.dump(self.feature_scaler, save_path / "feature_scaler.pkl")
            
            logger.info("Saved model artifacts")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {str(e)}")
            raise
    
    def _calculate_confidence(self, df: pd.DataFrame, predicted_price: float, current_price: float, 
                               volatility: float, avg_daily_change: float) -> tuple:
        """Calculate confidence score based on multiple market factors"""
        try:
            # 1. Calculate price change percentage (already in percentage form)
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [predicted_price, current_price, volatility, avg_daily_change]):
                logger.error("Invalid numeric inputs for confidence calculation")
                return self._get_default_confidence()
            
            if any(pd.isna([predicted_price, current_price, volatility, avg_daily_change])):
                logger.error("NaN values in confidence calculation inputs")
                return self._get_default_confidence()
            
            # 2. Get market conditions with safe fallbacks
            try:
                rsi = min(100, max(0, float(df['rsi'].iloc[-1])))  # RSI is already 0-100
            except (KeyError, IndexError, ValueError):
                rsi = 50  # Neutral RSI as fallback
                
            try:
                macd = float(df['macd'].iloc[-1])
                if not -100 <= macd <= 100:  # Sanity check for MACD
                    macd = 0
            except (KeyError, IndexError, ValueError):
                macd = 0  # Neutral MACD as fallback
                
            # Base confidence starts at 50%
            base_confidence = 50
            
            # 1. Price Change Factor (0 to -20 points)
            # More extreme predictions reduce confidence
            normal_daily_range = max(0.5, min(5, avg_daily_change))  # avg_daily_change is already in percentage
            change_penalty = min(20, max(0, (abs(price_change_pct) / normal_daily_range)))
            
            # 2. Volatility Factor (0 to -15 points)
            # Higher volatility reduces confidence
            normal_volatility = 20  # 20% annual volatility as baseline
            # volatility is already in percentage form
            volatility_penalty = min(15, max(0, (volatility - normal_volatility) / 4))
            
            # 3. Technical Indicators Factor (-10 to +10 points)
            tech_score = 0
            
            # RSI contribution (-5 to +5)
            if 45 <= rsi <= 55:  # Very neutral RSI
                tech_score += 5
            elif 35 <= rsi <= 65:  # Moderately neutral
                tech_score += 2
            elif rsi < 20 or rsi > 80:  # Extreme RSI
                tech_score -= 5
            
            # MACD contribution (-5 to +5)
            macd_range = avg_daily_change * 0.5  # avg_daily_change is already in percentage
            if abs(macd) < macd_range:  # Small MACD divergence
                tech_score += 5
            elif abs(macd) < macd_range * 2:  # Moderate divergence
                tech_score += 2
            else:  # Large divergence
                tech_score -= 5
            
            # Calculate final confidence
            confidence = base_confidence - change_penalty - volatility_penalty + tech_score
            
            # Ensure confidence stays within 0-100 range
            confidence = min(95, max(5, float(confidence)))  # Cap at 95% to account for market uncertainty
            
            # Log intermediate values for debugging
            logger.info(f"Confidence Calculation Breakdown:")
            logger.info(f"Base confidence: {base_confidence}")
            logger.info(f"Price change: {price_change_pct:.2f}%")
            logger.info(f"Normal daily range: {normal_daily_range:.2f}%")
            logger.info(f"Raw change penalty: {(abs(price_change_pct) / normal_daily_range):.2f}")
            logger.info(f"Final change penalty: {change_penalty:.1f}")
            logger.info(f"Volatility: {volatility:.1f}%")
            logger.info(f"Volatility penalty: {volatility_penalty:.1f}")
            logger.info(f"Technical score: {tech_score:+.1f}")
            logger.info(f"Final confidence: {confidence:.1f}%")
            
            confidence_details = {
                "base_confidence": float(base_confidence),
                "change_penalty": float(change_penalty),
                "volatility_penalty": float(volatility_penalty),
                "technical_score": float(tech_score),
                "price_change_pct": float(price_change_pct),
                "rsi": float(rsi),
                "macd": float(macd)
            }
            
            return float(confidence), confidence_details
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return self._get_default_confidence()
    
    def _get_default_confidence(self) -> tuple:
        """Return default confidence values"""
        return 50.0, {
            "base_confidence": 50,
            "change_penalty": 0,
            "volatility_penalty": 0,
            "technical_score": 0,
            "price_change_pct": 0,
            "rsi": 50,
            "macd": 0
        }

    def predict(self, days_ago: int = 0) -> Optional[Dict[str, Any]]:
        """Make a prediction using the trained model"""
        try:
            # Get end date based on days_ago
            end_date = datetime.now() - timedelta(days=days_ago)
            start_date = end_date - timedelta(days=60)  # Get 60 days of data for prediction
            
            # Get historical data
            df = self.data_service.get_historical_data_range(start_date, end_date)
            if df is None or df.empty:
                logger.warning("No historical data available for prediction")
                return None
            
            # Get current price and ensure it's a float
            try:
                current_price = float(df['close'].iloc[-1])
                logger.info(f"Current price: {current_price}")
            except (IndexError, ValueError) as e:
                logger.error(f"Error getting current price: {str(e)}")
                return None
            
            # Calculate historical volatility and average daily change
            daily_returns = df['close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility in percentage
            avg_daily_change = abs(daily_returns).mean() * 100  # Average daily change in percentage
            
            # Prepare data for prediction
            X, _, _ = self.prepare_data(df, settings.SEQUENCE_LENGTH)
            if X is None:
                logger.warning("Failed to prepare data for prediction")
                return None
                
            # Get the last sequence for prediction
            last_sequence = X[-1:]
            
            # Make prediction
            try:
                scaled_prediction = self.lstm_model.predict(last_sequence, verbose=0)[0][0]
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                return None
            
            # Get the last actual prices for inverse scaling
            last_prices = df[['open', 'high', 'low', 'close']].iloc[-1:]
            
            # Create a dummy row with the predicted close price
            dummy_row = last_prices.copy()
            dummy_row['close'] = scaled_prediction
            
            # Inverse transform the prediction
            predicted_price = float(self.price_scaler.inverse_transform(dummy_row)[0][3])
            
            # Calculate trend
            price_change = predicted_price - current_price
            trend = "up" if price_change > 0 else "down"
            
            # Calculate confidence and get details
            confidence, confidence_details = self._calculate_confidence(
                df, predicted_price, current_price, volatility, avg_daily_change
            )
            
            # Convert confidence from percentage (0-100) to decimal (0-1)
            confidence = confidence / 100.0
            
            # Calculate price change percentage with bounds
            price_change_pct = float(confidence_details["price_change_pct"])
            price_change_pct = min(100, max(-100, price_change_pct))  # Limit to ±100%
            
            # Convert percentage metrics to decimals for UI
            metrics = {
                "daily_volatility": float(volatility) / 100.0,  # Convert to decimal
                "avg_daily_change": float(avg_daily_change) / 100.0,  # Convert to decimal
                "rsi": float(df['rsi'].iloc[-1]) / 100.0,  # Convert to decimal
                "volume": int(df['volume'].iloc[-1]),
                "confidence_details": {
                    "base_confidence": float(confidence_details["base_confidence"]) / 100.0,
                    "change_penalty": float(confidence_details["change_penalty"]) / 100.0,
                    "volatility_penalty": float(confidence_details["volatility_penalty"]) / 100.0,
                    "technical_score": float(confidence_details["technical_score"]) / 100.0,
                    "price_change_pct": price_change_pct / 100.0,  # Convert to decimal
                    "rsi": float(confidence_details["rsi"]) / 100.0,
                    "macd": float(confidence_details["macd"]) / 100.0
                }
            }
            
            prediction_result = {
                "predicted_price": float(predicted_price),
                "actual_price": float(current_price),
                "trend": trend,
                "confidence": confidence,  # Now in decimal form (0.05 to 0.95)
                "last_close": float(current_price),
                "timestamp": end_date.isoformat(),
                "price_change_pct": price_change_pct / 100.0,  # Convert to decimal
                "metrics": metrics
            }
            
            # Log the final confidence value for debugging
            logger.info(f"Final prediction confidence: {confidence:.3f} ({confidence * 100:.1f}%)")
            logger.info(f"Price change percentage: {price_change_pct / 100:.3f} ({price_change_pct:.1f}%)")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in predict method: {str(e)}")
            return None
    
    async def start_training(self):
        """Start continuous training"""
        if self.training_task is None or self.training_task.done():
            self.training_task = asyncio.create_task(self.train_continuously())
    
    async def stop_training(self):
        """Stop continuous training"""
        if self.training_task and not self.training_task.done():
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                pass 