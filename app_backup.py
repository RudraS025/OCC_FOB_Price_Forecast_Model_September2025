"""
Modern Flask Web Application for OCC Price Forecasting
State-of-the-art ML-based forecasting with dynamic features
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging
import traceback

# Import our custom modules
from data_manager import DataManager
from forecasting_engine import ForecastingEngine
from simple_chart import create_simple_forecast_chart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'occ-price-forecasting-2025')

# Global instances
data_manager = None
forecasting_engine = None
viz_engine = None

def initialize_app():
    """Initialize the application components"""
    global data_manager, forecasting_engine, viz_engine
    
    try:
        # Initialize components
        data_manager = DataManager()
        forecasting_engine = ForecastingEngine()
        viz_engine = VisualizationEngine()
        
        # Train initial models
        logger.info("Training initial forecasting models...")
        data = data_manager.get_data()
        training_results = forecasting_engine.train_all_models(data)
        
        logger.info(f"Initialization complete. Best model: {training_results.get('best_model', 'Unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing app: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get data summary
        data_summary = data_manager.get_data_summary()
        
        # Get model information
        model_info = forecasting_engine.get_model_info()
        
        # Get recent data for display
        recent_data = data_manager.get_data().tail(12).copy()
        # Convert datetime to string for JSON serialization
        recent_data['Month'] = recent_data['Month'].dt.strftime('%Y-%m-%d')
        recent_data_json = recent_data.to_dict('records')
        
        return render_template('index.html', 
                             data_summary=data_summary,
                             model_info=model_info,
                             recent_data=recent_data_json)
        
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    """Forecast page with interactive forecasting"""
    if request.method == 'POST':
        try:
            # Get forecast parameters
            n_months = int(request.form.get('n_months', 6))
            model_name = request.form.get('model_name', None)
            
            # Validate input
            if n_months < 1 or n_months > 12:
                return jsonify({'error': 'Number of months must be between 1 and 12'}), 400
            
            # Generate forecast
            forecast_result = forecasting_engine.forecast(n_periods=n_months, model_name=model_name)
            
            if forecast_result['success']:
                # Get historical data for context
                historical_data = data_manager.get_data()
                
                # Create future dates
                future_dates = data_manager.generate_future_dates(n_months)
                future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
                
                # Prepare response
                response = {
                    'success': True,
                    'forecast': forecast_result['forecast'],
                    'lower_ci': forecast_result['lower_ci'],
                    'upper_ci': forecast_result['upper_ci'],
                    'future_dates': future_dates_str,
                    'model_used': forecast_result['model_used'],
                    'n_months': n_months,
                    'timestamp': datetime.now().isoformat()
                }
                
        # Generate visualization - simplified approach
        try:
            forecast_chart = create_simple_forecast_chart(
                historical_data, 
                forecast_result['forecast'], 
                future_dates,
                confidence_intervals=(forecast_result['lower_ci'], forecast_result['upper_ci'])
            )
            response['chart_html'] = forecast_chart
        except Exception as viz_error:
            logger.warning(f"Chart generation failed: {viz_error}")
            # Provide a simple fallback chart
            response['chart_html'] = f"""
            <div class="alert alert-info">
                <h5>Forecast Results</h5>
                <p>Forecast generated successfully for {n_months} months using {forecast_result['model_used']} model.</p>
                <p><strong>Predicted values:</strong> {', '.join([f'${v:.2f}' for v in forecast_result['forecast'][:3]])}{'...' if len(forecast_result['forecast']) > 3 else ''}</p>
                <p><em>Interactive chart temporarily unavailable - data is being processed correctly.</em></p>
            </div>
            """
                
                return jsonify(response)
            else:
                return jsonify({'error': forecast_result.get('error', 'Forecast generation failed')}), 500
                
        except Exception as e:
            logger.error(f"Error in forecast route: {e}")
            return jsonify({'error': str(e)}), 500
    
    # GET request - show forecast page
    try:
        model_info = forecasting_engine.get_model_info()
        return render_template('forecast.html', model_info=model_info)
    except Exception as e:
        logger.error(f"Error loading forecast page: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/update_data', methods=['GET', 'POST'])
def update_data():
    """Update actual price data"""
    if request.method == 'POST':
        try:
            # Get update parameters
            month = request.form.get('month')
            price = float(request.form.get('price'))
            
            # Validate input
            if not month or price <= 0:
                return jsonify({'error': 'Invalid month or price data'}), 400
            
            # Update data
            success = data_manager.add_actual_value(month, price)
            
            if success:
                # Retrain models with updated data
                logger.info("Retraining models with updated data...")
                updated_data = data_manager.get_data()
                training_results = forecasting_engine.train_all_models(updated_data)
                
                response = {
                    'success': True,
                    'message': f'Successfully updated price for {month} to ${price}',
                    'retrained_models': training_results['best_model'],
                    'data_points': len(updated_data)
                }
                
                return jsonify(response)
            else:
                return jsonify({'error': 'Failed to update data'}), 500
                
        except ValueError:
            return jsonify({'error': 'Invalid price value'}), 400
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return jsonify({'error': str(e)}), 500
    
    # GET request - show update page
    try:
        # Get next expected date
        next_date = data_manager.get_next_forecast_date()
        data_summary = data_manager.get_data_summary()
        
        return render_template('update_data.html', 
                             next_date=next_date.strftime('%Y-%m'),
                             data_summary=data_summary)
    except Exception as e:
        logger.error(f"Error loading update page: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/analytics')
def analytics():
    """Advanced analytics and model diagnostics"""
    try:
        # Get comprehensive data analysis
        data_summary = data_manager.get_data_summary()
        model_info = forecasting_engine.get_model_info()
        
        # Get historical data
        historical_data = data_manager.get_data()
        
        # Generate analytics charts
        charts = {
            'price_trend': viz_engine.create_price_trend_chart(historical_data),
            'seasonal_analysis': viz_engine.create_seasonal_analysis(historical_data),
            'model_performance': viz_engine.create_model_performance_chart(model_info),
            'data_statistics': viz_engine.create_statistics_dashboard(data_summary)
        }
        
        return render_template('analytics.html', 
                             data_summary=data_summary,
                             model_info=model_info,
                             charts=charts)
        
    except Exception as e:
        logger.error(f"Error in analytics route: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/api/data_summary')
def api_data_summary():
    """API endpoint for data summary"""
    try:
        summary = data_manager.get_data_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def api_model_info():
    """API endpoint for model information"""
    try:
        info = forecasting_engine.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical_data')
def api_historical_data():
    """API endpoint for historical data"""
    try:
        data = data_manager.get_data()
        data_dict = data.to_dict('records')
        return jsonify(data_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain_models', methods=['POST'])
def api_retrain_models():
    """API endpoint to manually retrain models"""
    try:
        data = data_manager.get_data()
        training_results = forecasting_engine.train_all_models(data)
        return jsonify(training_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    try:
        # Check if all components are working
        data_health = data_manager is not None
        model_health = forecasting_engine is not None and len(forecasting_engine.models) > 0
        
        return jsonify({
            'status': 'healthy' if data_health and model_health else 'unhealthy',
            'data_manager': 'ok' if data_health else 'error',
            'forecasting_engine': 'ok' if model_health else 'error',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error='Internal server error'), 500

# Template filters for better data display
@app.template_filter('datetime')
def datetime_filter(value):
    """Format datetime for templates"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime('%Y-%m-%d %H:%M')

@app.template_filter('currency')
def currency_filter(value):
    """Format currency for templates"""
    return f"${value:,.2f}"

@app.template_filter('percentage')
def percentage_filter(value):
    """Format percentage for templates"""
    return f"{value:.2f}%"

if __name__ == '__main__':
    # Initialize the application
    if initialize_app():
        # Run the application
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        app.run(host='0.0.0.0', port=port, debug=debug)
    else:
        print("Failed to initialize application. Exiting.")
        exit(1)
