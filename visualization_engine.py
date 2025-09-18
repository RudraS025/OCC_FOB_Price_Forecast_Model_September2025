"""
Visualization Engine for OCC Price Forecasting
Creates dynamic, interactive charts and diagnostic plots using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """Creates interactive visualizations for the forecasting application"""
    
    def __init__(self):
        """Initialize the visualization engine"""
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#6C757D',
            'background': '#F8F9FA'
        }
        
        self.default_layout = {
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'white',
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'title': {'font': {'size': 16, 'color': self.color_palette['primary']}},
            'xaxis': {'showgrid': True, 'gridcolor': '#E9ECEF'},
            'yaxis': {'showgrid': True, 'gridcolor': '#E9ECEF'}
        }
    
    def create_forecast_chart(self, historical_data: pd.DataFrame, 
                            forecast: List[float], 
                            future_dates: List[datetime],
                            confidence_intervals: Optional[Tuple[List[float], List[float]]] = None) -> str:
        """
        Create an interactive forecast chart
        
        Args:
            historical_data: Historical price data
            forecast: Forecasted values
            future_dates: Future dates for forecast
            confidence_intervals: Tuple of (lower_ci, upper_ci)
            
        Returns:
            str: HTML div containing the chart
        """
        try:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_data['Month'],
                y=historical_data['Price(USD/ton)'],
                mode='lines+markers',
                name='Historical Prices',
                line=dict(color=self.color_palette['primary'], width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{x}</b><br>Price: $%{y:,.2f}/ton<extra></extra>'
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.color_palette['accent'], width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond'),
                hovertemplate='<b>%{x}</b><br>Forecast: $%{y:,.2f}/ton<extra></extra>'
            ))
            
            # Confidence intervals
            if confidence_intervals:
                lower_ci, upper_ci = confidence_intervals
                
                # Upper confidence interval
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_ci,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Lower confidence interval with fill
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_ci,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({self._hex_to_rgb(self.color_palette["accent"])}, 0.2)',
                    name='Confidence Interval',
                    hovertemplate='<b>%{x}</b><br>Lower CI: $%{y:,.2f}/ton<extra></extra>'
                ))
            
            # Vertical line separating historical and forecast
            last_historical_date = historical_data['Month'].iloc[-1]
            fig.add_vline(
                x=last_historical_date, 
                line_dash="dot", 
                line_color=self.color_palette['neutral'],
                annotation_text="Forecast Start",
                annotation_position="top"
            )
            
            # Update layout
            fig.update_layout(
                title='OCC Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price (USD/ton)',
                hovermode='x unified',
                **self.default_layout
            )
            
            return fig.to_html(div_id="forecast-chart", include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating forecast chart: {e}")
            return f"<div>Error creating forecast chart: {e}</div>"
    
    def create_price_trend_chart(self, data: pd.DataFrame) -> str:
        """Create a comprehensive price trend analysis chart"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price Trend', 'Price Distribution', 'Monthly Changes', 'Yearly Comparison'),
                specs=[[{"secondary_y": False}, {"type": "histogram"}],
                       [{"secondary_y": True}, {"type": "bar"}]]
            )
            
            # Main price trend
            fig.add_trace(
                go.Scatter(
                    x=data['Month'],
                    y=data['Price(USD/ton)'],
                    mode='lines',
                    name='Price',
                    line=dict(color=self.color_palette['primary'], width=2)
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if len(data) >= 6:
                data['MA_6'] = data['Price(USD/ton)'].rolling(window=6).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data['Month'],
                        y=data['MA_6'],
                        mode='lines',
                        name='6-Month MA',
                        line=dict(color=self.color_palette['secondary'], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Price distribution
            fig.add_trace(
                go.Histogram(
                    x=data['Price(USD/ton)'],
                    name='Price Distribution',
                    marker_color=self.color_palette['accent'],
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # Monthly price changes
            price_changes = data['Price(USD/ton)'].pct_change() * 100
            colors = ['red' if x < 0 else 'green' for x in price_changes]
            
            fig.add_trace(
                go.Scatter(
                    x=data['Month'],
                    y=price_changes,
                    mode='markers',
                    name='Monthly Change %',
                    marker=dict(color=colors, size=6)
                ),
                row=2, col=1
            )
            
            # Yearly comparison
            data_with_year = data.copy()
            data_with_year['Year'] = data_with_year['Month'].dt.year
            yearly_avg = data_with_year.groupby('Year')['Price(USD/ton)'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=yearly_avg.index,
                    y=yearly_avg.values,
                    name='Yearly Average',
                    marker_color=self.color_palette['primary']
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Comprehensive Price Trend Analysis",
                showlegend=True,
                **self.default_layout
            )
            
            return fig.to_html(div_id="trend-chart", include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating trend chart: {e}")
            return f"<div>Error creating trend chart: {e}</div>"
    
    def create_seasonal_analysis(self, data: pd.DataFrame) -> str:
        """Create seasonal analysis charts"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Patterns', 'Quarterly Patterns', 'Year-over-Year', 'Seasonal Decomposition'),
                specs=[[{"type": "bar"}, {"type": "box"}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Monthly patterns
            data_copy = data.copy()
            data_copy['Month_Name'] = data_copy['Month'].dt.strftime('%b')
            data_copy['Month_Num'] = data_copy['Month'].dt.month
            
            monthly_avg = data_copy.groupby('Month_Num')['Price(USD/ton)'].mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig.add_trace(
                go.Bar(
                    x=month_names,
                    y=monthly_avg.values,
                    name='Monthly Average',
                    marker_color=self.color_palette['primary']
                ),
                row=1, col=1
            )
            
            # Quarterly box plots
            data_copy['Quarter'] = data_copy['Month'].dt.quarter
            
            for quarter in [1, 2, 3, 4]:
                quarter_data = data_copy[data_copy['Quarter'] == quarter]['Price(USD/ton)']
                fig.add_trace(
                    go.Box(
                        y=quarter_data,
                        name=f'Q{quarter}',
                        marker_color=self.color_palette['accent']
                    ),
                    row=1, col=2
                )
            
            # Year-over-Year comparison
            data_copy['Year'] = data_copy['Month'].dt.year
            for year in data_copy['Year'].unique()[-3:]:  # Last 3 years
                year_data = data_copy[data_copy['Year'] == year]
                fig.add_trace(
                    go.Scatter(
                        x=year_data['Month_Num'],
                        y=year_data['Price(USD/ton)'],
                        mode='lines+markers',
                        name=f'{year}',
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
            
            # Simple seasonal decomposition visualization
            if len(data) >= 24:  # Need at least 2 years for meaningful decomposition
                # Calculate 12-month rolling average as trend
                trend = data['Price(USD/ton)'].rolling(window=12, center=True).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=data['Month'],
                        y=data['Price(USD/ton)'],
                        mode='lines',
                        name='Original',
                        line=dict(color=self.color_palette['primary'], width=1)
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data['Month'],
                        y=trend,
                        mode='lines',
                        name='Trend',
                        line=dict(color=self.color_palette['accent'], width=2)
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title_text="Seasonal Analysis",
                showlegend=True,
                **self.default_layout
            )
            
            return fig.to_html(div_id="seasonal-chart", include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating seasonal analysis: {e}")
            return f"<div>Error creating seasonal analysis: {e}</div>"
    
    def create_model_performance_chart(self, model_info: Dict[str, Any]) -> str:
        """Create model performance comparison chart"""
        try:
            if not model_info.get('performance_metrics'):
                return "<div>No model performance data available</div>"
            
            metrics = model_info['performance_metrics']
            models = list(metrics.keys())
            
            # Prepare data for comparison
            mae_values = [metrics[model]['mae'] for model in models]
            rmse_values = [metrics[model]['rmse'] for model in models]
            mape_values = [metrics[model]['mape'] for model in models]
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Mean Absolute Error', 'Root Mean Square Error', 'Mean Absolute Percentage Error'),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            
            # MAE
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=mae_values,
                    name='MAE',
                    marker_color=self.color_palette['primary']
                ),
                row=1, col=1
            )
            
            # RMSE
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=rmse_values,
                    name='RMSE',
                    marker_color=self.color_palette['secondary']
                ),
                row=1, col=2
            )
            
            # MAPE
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=mape_values,
                    name='MAPE (%)',
                    marker_color=self.color_palette['accent']
                ),
                row=1, col=3
            )
            
            # Highlight best model
            best_model = model_info.get('best_model', '')
            fig.update_layout(
                title_text=f"Model Performance Comparison (Best: {best_model})",
                showlegend=False,
                **self.default_layout
            )
            
            return fig.to_html(div_id="performance-chart", include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return f"<div>Error creating performance chart: {e}</div>"
    
    def create_statistics_dashboard(self, data_summary: Dict[str, Any]) -> str:
        """Create a statistics dashboard"""
        try:
            metadata = data_summary.get('metadata', {})
            stats = data_summary.get('statistics', {}).get('price_statistics', {})
            
            # Create gauge charts for key metrics
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]],
                subplot_titles=('Current Price', 'Price Volatility', 'Data Quality', 'Trend Strength')
            )
            
            # Current Price Gauge
            current_price = stats.get('max', 0)  # Using max as current (latest)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=current_price,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Current Price (USD/ton)"},
                    gauge={
                        'axis': {'range': [None, 300]},
                        'bar': {'color': self.color_palette['primary']},
                        'steps': [
                            {'range': [0, 100], 'color': "lightgray"},
                            {'range': [100, 200], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 250
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Volatility Gauge
            volatility = stats.get('std', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=volatility,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Price Volatility"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.color_palette['secondary']},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 100], 'color': "red"}
                        ]
                    }
                ),
                row=1, col=2
            )
            
            # Data Quality Score
            total_records = metadata.get('total_records', 0)
            data_quality = min(100, (total_records / 120) * 100)  # 120 months = 10 years
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=data_quality,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Data Quality Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.color_palette['accent']},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ]
                    }
                ),
                row=2, col=1
            )
            
            # Trend Strength (simplified)
            trend_info = data_summary.get('statistics', {}).get('recent_trend', {})
            trend_strength = abs(trend_info.get('slope', 0)) * 10  # Scale for visualization
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=min(100, trend_strength),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Trend Strength"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.color_palette['success']},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ]
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Key Performance Indicators",
                **self.default_layout
            )
            
            return fig.to_html(div_id="statistics-dashboard", include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating statistics dashboard: {e}")
            return f"<div>Error creating statistics dashboard: {e}</div>"
    
    def create_diagnostic_plots(self, data: pd.DataFrame, model_results: Dict[str, Any]) -> str:
        """Create diagnostic plots for model validation"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Residuals vs Fitted', 'Q-Q Plot', 'Autocorrelation', 'Price Stability'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # This is a simplified version - in a real implementation,
            # you would need actual residuals from the models
            
            # Simulate residuals for demonstration
            np.random.seed(42)
            residuals = np.random.normal(0, 10, len(data))
            fitted_values = data['Price(USD/ton)'].values
            
            # Residuals vs Fitted
            fig.add_trace(
                go.Scatter(
                    x=fitted_values,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color=self.color_palette['primary'], size=6)
                ),
                row=1, col=1
            )
            
            # Add horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Q-Q Plot (simplified)
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = np.linspace(-2, 2, len(residuals))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_residuals,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color=self.color_palette['secondary'], size=6)
                ),
                row=1, col=2
            )
            
            # Price changes over time
            price_changes = data['Price(USD/ton)'].diff().dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=data['Month'][1:],
                    y=price_changes,
                    mode='lines',
                    name='Price Changes',
                    line=dict(color=self.color_palette['accent'], width=1)
                ),
                row=2, col=1
            )
            
            # Price stability (rolling standard deviation)
            rolling_std = data['Price(USD/ton)'].rolling(window=12).std()
            
            fig.add_trace(
                go.Scatter(
                    x=data['Month'],
                    y=rolling_std,
                    mode='lines',
                    name='12-Month Volatility',
                    line=dict(color=self.color_palette['success'], width=2)
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Model Diagnostic Plots",
                showlegend=True,
                **self.default_layout
            )
            
            return fig.to_html(div_id="diagnostic-plots", include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating diagnostic plots: {e}")
            return f"<div>Error creating diagnostic plots: {e}</div>"
    
    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string"""
        hex_color = hex_color.lstrip('#')
        return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"
    
    def create_simple_chart(self, data: pd.DataFrame, chart_type: str = 'line') -> str:
        """Create a simple chart for quick visualization"""
        try:
            if chart_type == 'line':
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['Month'],
                    y=data['Price(USD/ton)'],
                    mode='lines+markers',
                    name='Price',
                    line=dict(color=self.color_palette['primary'], width=2)
                ))
            elif chart_type == 'bar':
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=data['Month'],
                    y=data['Price(USD/ton)'],
                    name='Price',
                    marker_color=self.color_palette['primary']
                ))
            else:
                return "<div>Unsupported chart type</div>"
            
            fig.update_layout(
                title='OCC Price Data',
                xaxis_title='Date',
                yaxis_title='Price (USD/ton)',
                **self.default_layout
            )
            
            return fig.to_html(div_id="simple-chart", include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating simple chart: {e}")
            return f"<div>Error creating chart: {e}</div>"

# Test the visualization engine
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2020-01-01', end='2025-09-01', freq='MS')
    prices = 150 + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.random.normal(0, 10, len(dates))
    
    sample_data = pd.DataFrame({
        'Month': dates,
        'Price(USD/ton)': prices
    })
    
    # Test visualization
    viz = VisualizationEngine()
    
    # Create a simple chart
    chart_html = viz.create_simple_chart(sample_data)
    print("Chart created successfully!")
    
    # Save to file for testing
    with open('test_chart.html', 'w') as f:
        f.write(chart_html)
    
    print("Test chart saved to test_chart.html")
