import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forecasting_engine import ForecastingEngine
from data_manager import DataManager
import json

# Initialize components
data_manager = DataManager()
forecasting_engine = ForecastingEngine()

print("Loading data...")
data = data_manager.get_data()
print(f"Data shape: {data.shape}")

print("Training models...")
training_results = forecasting_engine.train_all_models(data)
print(f"Best model: {training_results['best_model']}")

print("Getting model info...")
model_info = forecasting_engine.get_model_info()
print(f"Available models: {model_info['available_models']}")
print(f"Best model: {model_info['best_model']}")

if model_info['performance_metrics']:
    print("Performance metrics structure:")
    for model_name, metrics in model_info['performance_metrics'].items():
        print(f"  {model_name}: {metrics}")

print("\nTesting forecast...")
forecast_result = forecasting_engine.forecast(n_periods=3)
print(f"Forecast result keys: {forecast_result.keys()}")
print(f"Success: {forecast_result.get('success', 'Not found')}")
if forecast_result.get('success'):
    print(f"Forecast: {forecast_result.get('forecast', 'Not found')}")
    print(f"Model used: {forecast_result.get('model_used', 'Not found')}")
else:
    print(f"Error: {forecast_result.get('error', 'Unknown error')}")
