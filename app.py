#!/usr/bin/env python3
"""
OCC Price Forecasting Web Application
A state-of-the-art ML-based web application for forecasting OCC prices
"""

import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import traceback

# Import custom modules
from data_manager import DataManager
from forecasting_engine import ForecastingEngine
from simple_chart import create_simple_forecast_chart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Global variables for app components
data_manager = None
forecasting_engine = None

def initialize_app():
    """Initialize the application components"""
    global data_manager, forecasting_engine
    
    try:
        # Initialize components
        data_manager = DataManager()
        forecasting_engine = ForecastingEngine()
        
        # Train initial models
        logger.info("Training initial forecasting models...")
        data = data_manager.get_data()
        training_results = forecasting_engine.train_all_models(data)
        
        logger.info(f"Initialization complete. Best model: {training_results.get('best_model', 'Unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
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
                
                # Generate visualization using simple chart
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
                    # Provide forecast data without chart
                    response['chart_html'] = f"""
                    <div class="alert alert-success">
                        <h5>✅ Forecast Generated Successfully!</h5>
                        <p><strong>Model:</strong> {forecast_result['model_used']}</p>
                        <p><strong>Forecast Period:</strong> {n_months} months</p>
                        <p><strong>Predicted Values:</strong></p>
                        <ul>
                            {''.join([f'<li>{future_dates_str[i]}: ${forecast_result["forecast"][i]:.2f}/ton</li>' for i in range(min(5, len(forecast_result["forecast"])))])}
                            {f'<li>... and {len(forecast_result["forecast"]) - 5} more values</li>' if len(forecast_result["forecast"]) > 5 else ''}
                        </ul>
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
            
            # Validate inputs
            if not month or price <= 0:
                return jsonify({'error': 'Invalid month or price value'}), 400
            
            # Update data
            success = data_manager.add_actual_value(month, price)
            
            if success:
                # Retrain models with updated data
                data = data_manager.get_data()
                training_results = forecasting_engine.train_all_models(data)
                
                return jsonify({
                    'success': True,
                    'message': f'Data updated for {month}',
                    'new_best_model': training_results.get('best_model', 'Unknown'),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Failed to update data'}), 500
                
        except Exception as e:
            logger.error(f"Error in update_data route: {e}")
            return jsonify({'error': str(e)}), 500
    
    # GET request - show update page
    try:
        recent_data = data_manager.get_data().tail(10)
        recent_data_json = recent_data.to_dict('records')
        data_summary = data_manager.get_data_summary()
        return render_template('update_data.html', 
                             recent_data=recent_data_json,
                             data_summary=data_summary)
    except Exception as e:
        logger.error(f"Error loading update_data page: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/analytics')
def analytics():
    """Modern Analytics Intelligence Center"""
    try:
        # Get model performance data
        model_info = forecasting_engine.get_model_info()
        
        # Get data validation results
        data_validation = data_manager.validate_data()
        
        # Get data summary
        data_summary = data_manager.get_data_summary()
        
        return render_template('analytics.html', 
                             model_info=model_info,
                             data_validation=data_validation,
                             data_summary=data_summary)
    except Exception as e:
        logger.error(f"Error in analytics route: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/api/historical_data')
def api_historical_data():
    """API endpoint for historical data"""
    try:
        data = data_manager.get_data()
        # Convert to JSON-serializable format
        data_copy = data.copy()
        data_copy['Month'] = data_copy['Month'].dt.strftime('%Y-%m-%d')
        return jsonify(data_copy.to_dict('records'))
    except Exception as e:
        logger.error(f"Error in historical_data API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_performance')
def api_model_performance():
    """API endpoint for model performance metrics"""
    try:
        performance_data = forecasting_engine.get_model_info()
        return jsonify(performance_data)
    except Exception as e:
        logger.error(f"Error in model_performance API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain_models', methods=['POST'])
def api_retrain_models():
    """API endpoint to retrain all models"""
    try:
        logger.info("Starting model retraining...")
        
        # Retrain all models
        forecasting_engine.train_models()
        
        # Get updated model info
        model_info = forecasting_engine.get_model_info()
        
        logger.info(f"Model retraining completed. Best model: {model_info['best_model']}")
        
        return jsonify({
            'success': True,
            'best_model': model_info['best_model'],
            'available_models': model_info['available_models'],
            'performance_metrics': model_info['performance_metrics'],
            'message': 'All models retrained successfully'
        })
        
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrain models'
        }), 500

@app.route('/api/data_summary')
def api_data_summary():
    """API endpoint for data summary"""
    try:
        data_summary = data_manager.get_data_summary()
        return jsonify(data_summary)
    except Exception as e:
        logger.error(f"Error in data_summary API: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Initialize the application
    if initialize_app():
        logger.info("Starting OCC Price Forecasting Application...")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize application. Exiting.")
        exit(1)
