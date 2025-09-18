"""
Lightweight Flask Application for Heroku Deployment
OCC FOB Price Forecasting with ML Models
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Use lightweight forecasting engine for Heroku
try:
    from forecasting_engine_heroku import ForecastingEngine
except ImportError:
    from forecasting_engine import ForecastingEngine

from data_manager import DataManager
from visualization_engine import VisualizationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'occ-price-forecasting-2025')

# Initialize components
data_manager = DataManager()
forecasting_engine = ForecastingEngine()
visualization_engine = VisualizationEngine()

# Global initialization flag
initialized = False

def initialize_app():
    """Initialize the application with data and models"""
    global initialized
    
    if initialized:
        return
    
    try:
        logger.info("Training initial forecasting models...")
        
        # Get data and train models
        data = data_manager.get_data()
        training_results = forecasting_engine.train_all_models(data)
        
        logger.info(f"Initialization complete. Best model: {training_results.get('best_model', 'Unknown')}")
        initialized = True
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        # Continue with limited functionality

@app.before_first_request
def startup():
    """Initialize the app on first request"""
    initialize_app()

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Ensure initialization
        if not initialized:
            initialize_app()
        
        # Get data summary
        data_summary = data_manager.get_data_summary()
        
        # Get model information
        model_info = forecasting_engine.get_model_info()
        
        # Get recent data for display
        recent_data = data_manager.get_data().tail(12).copy()
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
            # Ensure initialization
            if not initialized:
                initialize_app()
            
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
                
                # Prepare response data
                forecast_data = []
                for i, date in enumerate(future_dates):
                    forecast_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'forecast': round(forecast_result['forecast'][i], 2),
                        'lower_ci': round(forecast_result['lower_ci'][i], 2),
                        'upper_ci': round(forecast_result['upper_ci'][i], 2)
                    })
                
                return jsonify({
                    'success': True,
                    'model_used': forecast_result['model_used'],
                    'forecast_data': forecast_data,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Forecast generation failed'}), 500
                
        except Exception as e:
            logger.error(f"Error in forecast route: {e}")
            return jsonify({'error': str(e)}), 500
    
    # GET request - show forecast page
    try:
        # Ensure initialization
        if not initialized:
            initialize_app()
        
        model_info = forecasting_engine.get_model_info()
        data_summary = data_manager.get_data_summary()
        return render_template('forecast.html', 
                             model_info=model_info,
                             data_summary=data_summary)
    except Exception as e:
        logger.error(f"Error loading forecast page: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/update_data', methods=['GET', 'POST'])
def update_data():
    """Update data with new actual values"""
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

# API Routes
@app.route('/api/historical_data')
def api_historical_data():
    """API endpoint for historical data"""
    try:
        data = data_manager.get_data()
        data_copy = data.copy()
        data_copy['Month'] = data_copy['Month'].dt.strftime('%Y-%m-%d')
        return jsonify(data_copy.to_dict('records'))
    except Exception as e:
        logger.error(f"Error in api_historical_data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data_summary')
def api_data_summary():
    """API endpoint for data summary"""
    try:
        summary = data_manager.get_data_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error in api_data_summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/table_data')
def api_table_data():
    """API endpoint for table data with pagination"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        data = data_manager.get_data()
        
        # Calculate pagination
        total = len(data)
        start = (page - 1) * per_page
        end = start + per_page
        
        paginated_data = data.iloc[start:end].copy()
        paginated_data['Month'] = paginated_data['Month'].dt.strftime('%Y-%m-%d')
        
        return jsonify({
            'data': paginated_data.to_dict('records'),
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
    except Exception as e:
        logger.error(f"Error in api_table_data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'initialized': initialized,
        'models_available': list(forecasting_engine.models.keys()) if initialized else []
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    logger.info("Starting OCC Price Forecasting Application...")
    
    # Initialize the application
    initialize_app()
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
