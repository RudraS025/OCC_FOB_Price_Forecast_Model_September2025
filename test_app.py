#!/usr/bin/env python3
"""
Test script for OCC Price Forecasting Application
"""

import sys
import os
import pandas as pd
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_manager import DataManager
from forecasting_engine import ForecastingEngine
from visualization_engine import VisualizationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading functionality"""
    logger.info("Testing data loading...")
    
    try:
        data_manager = DataManager('US_OCC_PRICES_Sep25.xlsx', 'occ_price_data.csv')
        data = data_manager.get_data()
        
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Data types: {data.dtypes}")
        logger.info(f"Date range: {data['Month'].min()} to {data['Month'].max()}")
        
        return True
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return False

def test_forecasting():
    """Test forecasting functionality"""
    logger.info("Testing forecasting...")
    
    try:
        data_manager = DataManager('US_OCC_PRICES_Sep25.xlsx', 'occ_price_data.csv')
        forecasting_engine = ForecastingEngine()
        
        # Train models
        logger.info("Training models...")
        forecasting_engine.train_all_models(data_manager.get_data())
        
        # Generate forecast
        logger.info("Generating forecast...")
        forecast_result = forecasting_engine.forecast(6)
        
        if forecast_result['success']:
            logger.info(f"Forecast successful with model: {forecast_result['model_used']}")
            logger.info(f"Forecast values: {forecast_result['forecast'][:3]}...")  # Show first 3
            return True
        else:
            logger.error(f"Forecast failed: {forecast_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Forecasting test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality"""
    logger.info("Testing visualization...")
    
    try:
        data_manager = DataManager('US_OCC_PRICES_Sep25.xlsx', 'occ_price_data.csv')
        forecasting_engine = ForecastingEngine()
        viz_engine = VisualizationEngine()
        
        # Get data and train models
        data = data_manager.get_data()
        forecasting_engine.train_all_models(data)
        
        # Generate forecast
        forecast_result = forecasting_engine.forecast(6)
        
        if forecast_result['success']:
            # Generate future dates
            future_dates = data_manager.generate_future_dates(6)
            
            # Create chart
            chart_html = viz_engine.create_forecast_chart(
                data,
                forecast_result['forecast'],
                future_dates,
                confidence_intervals=(forecast_result['lower_ci'], forecast_result['upper_ci'])
            )
            
            logger.info("Visualization test successful")
            return True
        else:
            logger.error("Forecast failed in visualization test")
            return False
            
    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
        return False

def test_json_serialization():
    """Test JSON serialization of data"""
    logger.info("Testing JSON serialization...")
    
    try:
        data_manager = DataManager('US_OCC_PRICES_Sep25.xlsx', 'occ_price_data.csv')
        data = data_manager.get_data().tail(5).copy()
        
        # Convert datetime to string for JSON serialization
        data['Month'] = data['Month'].dt.strftime('%Y-%m-%d')
        data_json = data.to_dict('records')
        
        logger.info(f"JSON serialization successful: {len(data_json)} records")
        logger.info(f"Sample record: {data_json[0]}")
        return True
        
    except Exception as e:
        logger.error(f"JSON serialization test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting OCC Price Forecasting Application Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Forecasting", test_forecasting),
        ("JSON Serialization", test_json_serialization),
        ("Visualization", test_visualization)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        results[test_name] = test_func()
        status = "PASSED" if results[test_name] else "FAILED"
        logger.info(f"{test_name} test: {status}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Application is ready for deployment.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
