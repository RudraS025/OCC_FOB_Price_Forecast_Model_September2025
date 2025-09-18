# OCC Price Forecasting System

A state-of-the-art machine learning-based web application for forecasting Old Corrugated Cardboard (OCC) prices. This application provides accurate price predictions using multiple ML models, dynamic data updates, and interactive visualizations.

## 🌟 Features

### Core Functionality
- **Advanced ML Forecasting**: Multiple models including ARIMA, LSTM, Prophet, Exponential Smoothing, and Random Forest
- **Dynamic Data Updates**: Real-time data addition with automatic model retraining
- **Interactive Visualizations**: Plotly-powered charts and dashboards
- **Flexible Forecast Periods**: Generate forecasts for 1-12 months ahead
- **Model Performance Comparison**: Automatic best model selection with performance metrics

### Technical Features
- **Responsive Web Design**: Modern Bootstrap-based UI
- **RESTful API**: JSON endpoints for data access
- **Cloud-Ready**: Optimized for Heroku deployment
- **Data Validation**: Comprehensive data quality checks
- **Export Capabilities**: Download data in multiple formats

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- Heroku CLI (for deployment)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/occ-price-forecasting.git
cd occ-price-forecasting
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Initialize the application**
```bash
python app.py
```

4. **Access the application**
Open your browser and navigate to `http://localhost:5000`

## 📊 Data Structure

The application expects Excel data with the following structure:
- **Month**: Date column (YYYY-MM-DD format)
- **Price(USD/ton)**: Numeric price values

Example:
```
Month        | Price(USD/ton)
2017-01-01   | 179
2017-02-01   | 202
...
```

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```env
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DEBUG=False
PORT=5000
```

### Model Configuration
The application includes pre-configured ML models:
- **ARIMA**: Classical time series analysis
- **LSTM**: Deep learning neural networks  
- **Prophet**: Facebook's forecasting tool
- **Exponential Smoothing**: Trend and seasonality modeling
- **Random Forest**: Ensemble learning approach

## 🌐 Deployment

### Heroku Deployment

1. **Create Heroku app**
```bash
heroku create your-app-name
```

2. **Set environment variables**
```bash
heroku config:set SECRET_KEY=your-secret-key
heroku config:set FLASK_ENV=production
```

3. **Deploy**
```bash
git add .
git commit -m "Initial deployment"
git push heroku main
```

4. **Open the application**
```bash
heroku open
```

### Manual Deployment (Other Platforms)

1. **Build the application**
```bash
pip install -r requirements.txt
```

2. **Set environment variables**
```bash
export FLASK_APP=app.py
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
```

3. **Run with Gunicorn**
```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

## 📱 Usage Guide

### Dashboard
- View recent price data and trends
- Monitor system status and model performance
- Generate quick forecasts

### Forecasting
1. Select forecast period (1-12 months)
2. Choose ML model (or use auto-selection)
3. Click "Generate Forecast"
4. View results with confidence intervals

### Data Updates
1. Navigate to "Update Data"
2. Select month and enter price
3. Submit to automatically retrain models
4. View updated statistics

### Analytics
- Comprehensive trend analysis
- Seasonal pattern detection
- Model performance comparison
- Data quality reports

## 🔌 API Endpoints

### Data Endpoints
- `GET /api/data_summary` - Dataset summary and metadata
- `GET /api/historical_data` - Complete historical dataset
- `GET /api/model_info` - Model information and performance

### Functionality Endpoints
- `POST /forecast` - Generate price forecasts
- `POST /update_data` - Add new price data
- `POST /api/retrain_models` - Manually retrain models

### Health Check
- `GET /health` - Application health status

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Manual Testing
1. Start the application locally
2. Upload sample data
3. Generate forecasts
4. Update data points
5. Verify model retraining

## 📈 Model Performance

The application automatically evaluates models using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)

Best performing model is automatically selected for forecasting.

## 🔒 Security Features

- CSRF protection
- Input validation
- Session management
- Environment-based configuration
- Secure headers

## 🛠️ Development

### Project Structure
```
occ-price-forecasting/
├── app.py                 # Main Flask application
├── data_manager.py        # Data handling and validation
├── forecasting_engine.py  # ML models and forecasting
├── visualization_engine.py # Charts and visualizations
├── templates/             # HTML templates
├── static/               # CSS, JS, and assets
├── requirements.txt      # Python dependencies
├── Procfile             # Heroku deployment
└── README.md            # This file
```

### Adding New Models
1. Implement model in `forecasting_engine.py`
2. Add training method
3. Update model evaluation
4. Test performance

### Customizing UI
- Modify templates in `templates/`
- Update styles in `static/css/style.css`
- Add functionality in `static/js/app.js`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**Application won't start**
- Check Python version (3.11+ required)
- Verify all dependencies are installed
- Check environment variables

**Forecast generation fails**
- Ensure data has at least 24 months
- Check data format and quality
- Verify model training completed

**Charts not displaying**
- Check internet connection (CDN dependencies)
- Verify JavaScript console for errors
- Ensure Plotly is loading correctly

### Getting Help
- Check the GitHub Issues page
- Review the application logs
- Contact support team

## 🚀 Future Enhancements

- [ ] Real-time data streaming
- [ ] Advanced ensemble models
- [ ] Email alert system
- [ ] Multi-commodity support
- [ ] Advanced analytics dashboard
- [ ] Mobile application
- [ ] API authentication
- [ ] Data backup system

## 📊 Performance Benchmarks

Typical performance metrics:
- **Model Training**: 30-60 seconds
- **Forecast Generation**: 1-3 seconds
- **Data Update**: 2-5 seconds
- **Page Load**: <2 seconds

## 🏆 Acknowledgments

- Facebook Prophet for time series forecasting
- Plotly for interactive visualizations
- Bootstrap for responsive design
- Flask community for web framework
- Heroku for cloud hosting

---

**Built with ❤️ for accurate OCC price forecasting**

For questions or support, please open an issue on GitHub or contact the development team.
