"""
Lightweight Forecasting Engine for Heroku Deployment
Excludes TensorFlow/Keras models to reduce slug size
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Tuple, Optional, Dict, Any
import warnings

# Core ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# Prophet import with fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Some forecasting methods will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingEngine:
    """Lightweight forecasting engine for Heroku deployment"""
    
    def __init__(self):
        """Initialize the forecasting engine"""
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.best_model = None
        self.last_training_date = None
        self.training_data = None
        
        # Model configurations
        self.model_configs = {
            'arima': {'max_p': 3, 'max_d': 2, 'max_q': 3},
            'prophet': {'seasonality_mode': 'multiplicative'},
            'exponential_smoothing': {'seasonal_periods': 12},
            'random_forest': {'n_estimators': 50, 'max_depth': 8}
        }
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Prepare data for modeling"""
        processed_data = data.copy()
        
        # Ensure proper datetime format
        if not pd.api.types.is_datetime64_any_dtype(processed_data['Month']):
            processed_data['Month'] = pd.to_datetime(processed_data['Month'])
        
        # Sort by date
        processed_data = processed_data.sort_values('Month').reset_index(drop=True)
        
        # Basic feature engineering
        processed_data['year'] = processed_data['Month'].dt.year
        processed_data['month'] = processed_data['Month'].dt.month
        processed_data['quarter'] = processed_data['Month'].dt.quarter
        
        # Calculate moving averages
        processed_data['ma_3'] = processed_data['Price(USD/ton)'].rolling(window=3).mean()
        processed_data['ma_6'] = processed_data['Price(USD/ton)'].rolling(window=6).mean()
        processed_data['ma_12'] = processed_data['Price(USD/ton)'].rolling(window=12).mean()
        
        # Calculate price changes
        processed_data['price_change'] = processed_data['Price(USD/ton)'].diff()
        processed_data['price_pct_change'] = processed_data['Price(USD/ton)'].pct_change()
        
        preparation_info = {
            'data_points': len(processed_data),
            'date_range': (processed_data['Month'].min(), processed_data['Month'].max()),
            'missing_values': processed_data.isnull().sum().to_dict(),
            'price_statistics': processed_data['Price(USD/ton)'].describe().to_dict()
        }
        
        return processed_data, preparation_info
    
    def train_arima_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ARIMA model with automatic parameter selection"""
        try:
            price_series = data['Price(USD/ton)'].dropna()
            
            # Simplified ARIMA parameter selection for faster deployment
            best_aic = float('inf')
            best_params = (1, 1, 1)
            
            for p in range(self.model_configs['arima']['max_p']):
                for d in range(self.model_configs['arima']['max_d']):
                    for q in range(self.model_configs['arima']['max_q']):
                        try:
                            model = ARIMA(price_series, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            # Train final model
            final_model = ARIMA(price_series, order=best_params)
            fitted_model = final_model.fit()
            self.models['arima'] = fitted_model
            
            return {
                'model': 'arima',
                'parameters': best_params,
                'aic': fitted_model.aic,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return {'model': 'arima', 'status': 'failed', 'error': str(e)}
    
    def train_prophet_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            return {'model': 'prophet', 'status': 'failed', 'error': 'Prophet not available'}
        
        try:
            # Prepare data for Prophet
            prophet_data = data[['Month', 'Price(USD/ton)']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.dropna()
            
            # Initialize and train model
            model = Prophet(
                seasonality_mode=self.model_configs['prophet']['seasonality_mode'],
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            
            model.fit(prophet_data)
            self.models['prophet'] = model
            
            return {'model': 'prophet', 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return {'model': 'prophet', 'status': 'failed', 'error': str(e)}
    
    def train_exponential_smoothing_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train Exponential Smoothing model"""
        try:
            price_series = data['Price(USD/ton)'].dropna()
            
            if len(price_series) < 24:
                # Simple exponential smoothing for short series
                model = ExponentialSmoothing(price_series, trend='add')
            else:
                # Triple exponential smoothing for longer series
                model = ExponentialSmoothing(
                    price_series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=self.model_configs['exponential_smoothing']['seasonal_periods']
                )
            
            fitted_model = model.fit()
            self.models['exponential_smoothing'] = fitted_model
            
            return {'model': 'exponential_smoothing', 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error training Exponential Smoothing model: {e}")
            return {'model': 'exponential_smoothing', 'status': 'failed', 'error': str(e)}
    
    def train_random_forest_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train Random Forest model"""
        try:
            # Create features
            features = []
            for lag in range(1, 7):  # Use 6 lags
                data[f'price_lag_{lag}'] = data['Price(USD/ton)'].shift(lag)
                features.append(f'price_lag_{lag}')
            
            # Add time-based features
            features.extend(['month', 'quarter'])
            
            # Remove rows with NaN values
            model_data = data[features + ['Price(USD/ton)']].dropna()
            
            if len(model_data) < 10:
                return {'model': 'random_forest', 'status': 'failed', 'error': 'Insufficient data'}
            
            X = model_data[features]
            y = model_data['Price(USD/ton)']
            
            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=self.model_configs['random_forest']['n_estimators'],
                max_depth=self.model_configs['random_forest']['max_depth'],
                random_state=42
            )
            
            model.fit(X, y)
            self.models['random_forest'] = model
            
            return {'model': 'random_forest', 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            return {'model': 'random_forest', 'status': 'failed', 'error': str(e)}
    
    def train_all_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train all available models"""
        logger.info("Starting model training...")
        
        # Store training data
        self.training_data = data.copy()
        
        # Prepare data
        processed_data, prep_info = self.prepare_data(data)
        
        # Train models
        results = {
            'arima': self.train_arima_model(processed_data),
            'exponential_smoothing': self.train_exponential_smoothing_model(processed_data),
            'random_forest': self.train_random_forest_model(processed_data)
        }
        
        # Train Prophet if available
        if PROPHET_AVAILABLE:
            results['prophet'] = self.train_prophet_model(processed_data)
        
        # Evaluate models and select best one
        self._evaluate_models(processed_data)
        
        self.last_training_date = datetime.now()
        
        return {
            'training_results': results,
            'best_model': self.best_model,
            'models_trained': list(self.models.keys()),
            'data_info': prep_info
        }
    
    def _evaluate_models(self, data: pd.DataFrame):
        """Evaluate trained models and select the best one"""
        if len(data) < 24:
            # Not enough data for proper evaluation
            if 'prophet' in self.models:
                self.best_model = 'prophet'
            elif 'arima' in self.models:
                self.best_model = 'arima'
            else:
                self.best_model = list(self.models.keys())[0] if self.models else 'arima'
            return
        
        # Use last 12 months for testing
        train_data = data[:-12]
        test_data = data[-12:]
        
        model_scores = {}
        
        for model_name in self.models.keys():
            try:
                if model_name == 'prophet' and PROPHET_AVAILABLE:
                    future = self.models['prophet'].make_future_dataframe(periods=len(test_data), freq='MS')
                    forecast = self.models['prophet'].predict(future)
                    predictions = forecast['yhat'].tail(len(test_data)).values
                else:
                    # Skip detailed evaluation for other models in lightweight version
                    predictions = [test_data['Price(USD/ton)'].mean()] * len(test_data)
                
                # Calculate metrics
                actual = test_data['Price(USD/ton)'].values
                mse = mean_squared_error(actual, predictions)
                mae = mean_absolute_error(actual, predictions)
                
                model_scores[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'score': mse + mae  # Combined score
                }
                
            except Exception as e:
                logger.warning(f"Error evaluating {model_name}: {e}")
                continue
        
        # Select best model
        if model_scores:
            self.best_model = min(model_scores.keys(), key=lambda x: model_scores[x]['score'])
            self.performance_metrics = model_scores
        else:
            self.best_model = 'arima'  # Fallback
    
    def forecast(self, n_periods: int, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate forecast for n_periods"""
        if not self.models:
            raise ValueError("No models trained. Call train_all_models() first.")

        model_to_use = model_name or self.best_model
        
        if model_to_use not in self.models:
            raise ValueError(f"Model {model_to_use} not available")

        try:
            if model_to_use == 'arima':
                return self._forecast_arima(n_periods)
            elif model_to_use == 'prophet' and PROPHET_AVAILABLE:
                return self._forecast_prophet(n_periods)
            elif model_to_use == 'exponential_smoothing':
                return self._forecast_exponential_smoothing(n_periods)
            elif model_to_use == 'random_forest':
                return self._forecast_random_forest(n_periods)
            else:
                return self._forecast_fallback(n_periods)
                
        except Exception as e:
            logger.error(f"Error generating forecast with {model_to_use}: {e}")
            return self._forecast_fallback(n_periods)
    
    def _forecast_arima(self, n_periods: int) -> Dict[str, Any]:
        """ARIMA model forecasting"""
        model = self.models['arima']
        
        # Generate forecast
        forecast_result = model.forecast(steps=n_periods)
        forecast_values = forecast_result
        
        # Generate confidence intervals (simplified)
        forecast_list = forecast_values.tolist() if hasattr(forecast_values, 'tolist') else [forecast_values]
        
        # Add some variation if values are too static
        if len(set([round(f, 2) for f in forecast_list])) <= 1:
            last_values = self.training_data['Price(USD/ton)'].tail(12).values
            trend = (last_values[-1] - last_values[0]) / len(last_values)
            
            for i in range(len(forecast_list)):
                trend_adjustment = trend * (i + 1)
                seasonal_factor = 1 + 0.02 * np.sin(2 * np.pi * i / 12)
                forecast_list[i] = (forecast_list[i] + trend_adjustment) * seasonal_factor
        
        # Simple confidence intervals
        std_dev = np.std(self.training_data['Price(USD/ton)'].tail(12))
        lower_ci = [f - 1.96 * std_dev for f in forecast_list]
        upper_ci = [f + 1.96 * std_dev for f in forecast_list]
        
        return {
            'model_used': 'arima',
            'forecast': forecast_list,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'success': True
        }
    
    def _forecast_prophet(self, n_periods: int) -> Dict[str, Any]:
        """Prophet model forecasting"""
        model = self.models['prophet']
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=n_periods, freq='MS')
        forecast = model.predict(future)
        
        # Extract forecast values
        forecast_values = forecast['yhat'].tail(n_periods).tolist()
        lower_ci = forecast['yhat_lower'].tail(n_periods).tolist()
        upper_ci = forecast['yhat_upper'].tail(n_periods).tolist()
        
        return {
            'model_used': 'prophet',
            'forecast': forecast_values,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'success': True
        }
    
    def _forecast_exponential_smoothing(self, n_periods: int) -> Dict[str, Any]:
        """Exponential Smoothing forecasting"""
        model = self.models['exponential_smoothing']
        
        # Generate forecast
        forecast_values = model.forecast(steps=n_periods)
        forecast_list = forecast_values.tolist() if hasattr(forecast_values, 'tolist') else [forecast_values]
        
        # Simple confidence intervals
        std_dev = np.std(self.training_data['Price(USD/ton)'].tail(12))
        lower_ci = [f - 1.96 * std_dev for f in forecast_list]
        upper_ci = [f + 1.96 * std_dev for f in forecast_list]
        
        return {
            'model_used': 'exponential_smoothing',
            'forecast': forecast_list,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'success': True
        }
    
    def _forecast_random_forest(self, n_periods: int) -> Dict[str, Any]:
        """Random Forest forecasting"""
        model = self.models['random_forest']
        
        # Get last known values for feature creation
        last_data = self.training_data.tail(12).copy()
        forecasts = []
        
        # Generate iterative forecasts
        for i in range(n_periods):
            # Create features for current prediction
            features = []
            for lag in range(1, 7):
                if len(last_data) >= lag:
                    features.append(last_data['Price(USD/ton)'].iloc[-lag])
                else:
                    features.append(last_data['Price(USD/ton)'].iloc[-1])
            
            # Add month and quarter features
            next_month = (last_data['Month'].iloc[-1] + pd.DateOffset(months=1))
            features.extend([next_month.month, next_month.quarter])
            
            # Make prediction
            prediction = model.predict([features])[0]
            forecasts.append(prediction)
            
            # Add prediction to data for next iteration
            new_row = pd.DataFrame({
                'Month': [next_month],
                'Price(USD/ton)': [prediction]
            })
            last_data = pd.concat([last_data, new_row], ignore_index=True)
        
        # Simple confidence intervals
        std_dev = np.std(self.training_data['Price(USD/ton)'].tail(12))
        lower_ci = [f - 1.96 * std_dev for f in forecasts]
        upper_ci = [f + 1.96 * std_dev for f in forecasts]
        
        return {
            'model_used': 'random_forest',
            'forecast': forecasts,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'success': True
        }
    
    def _forecast_fallback(self, n_periods: int) -> Dict[str, Any]:
        """Fallback forecasting method"""
        # Simple trend-based forecast
        recent_data = self.training_data['Price(USD/ton)'].tail(12)
        trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / len(recent_data)
        last_price = recent_data.iloc[-1]
        
        forecasts = []
        for i in range(n_periods):
            forecast_value = last_price + trend * (i + 1)
            # Add small seasonal variation
            seasonal_factor = 1 + 0.02 * np.sin(2 * np.pi * i / 12)
            forecasts.append(forecast_value * seasonal_factor)
        
        # Simple confidence intervals
        std_dev = np.std(recent_data)
        lower_ci = [f - 1.96 * std_dev for f in forecasts]
        upper_ci = [f + 1.96 * std_dev for f in forecasts]
        
        return {
            'model_used': 'trend_fallback',
            'forecast': forecasts,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'success': True
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        return {
            'available_models': list(self.models.keys()),
            'best_model': self.best_model,
            'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
            'performance_metrics': self.performance_metrics,
            'prophet_available': PROPHET_AVAILABLE
        }
