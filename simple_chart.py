#!/usr/bin/env python3
"""
Simple Forecast Chart Generator - Bypasses Timestamp Issues
"""

import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def create_simple_forecast_chart(historical_data: pd.DataFrame, 
                                forecast: List[float], 
                                future_dates: List,
                                confidence_intervals: Optional[Tuple[List[float], List[float]]] = None) -> str:
    """
    Create a simple forecast chart that avoids timestamp arithmetic issues
    """
    try:
        fig = go.Figure()
        
        # Convert dates to strings to avoid timestamp issues
        hist_dates = [str(d)[:10] for d in historical_data['Month']]
        future_dates_str = [str(d)[:10] if hasattr(d, 'strftime') else str(d)[:10] for d in future_dates]
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=historical_data['Price(USD/ton)'].tolist(),
            mode='lines+markers',
            name='Historical Prices',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=future_dates_str,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#A23B72', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Confidence intervals if available
        if confidence_intervals:
            lower_ci, upper_ci = confidence_intervals
            
            # Upper bound
            fig.add_trace(go.Scatter(
                x=future_dates_str,
                y=upper_ci,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=future_dates_str,
                y=lower_ci,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(162, 59, 114, 0.2)',
                name='Confidence Interval'
            ))
        
        # Update layout
        fig.update_layout(
            title='OCC Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price (USD/ton)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig.to_html(div_id="forecast-chart", include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Simple chart creation failed: {e}")
        return f"""
        <div class="alert alert-warning">
            <h5>Chart Generation Issue</h5>
            <p>Unable to generate interactive chart. Forecast data is available in the response.</p>
            <p>Error: {str(e)}</p>
        </div>
        """

if __name__ == "__main__":
    # Test the function
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=12, freq='MS')
    prices = 100 + np.random.randn(12) * 10
    
    hist_data = pd.DataFrame({
        'Month': dates,
        'Price(USD/ton)': prices
    })
    
    future_dates = pd.date_range('2025-01-01', periods=6, freq='MS')
    forecast_values = [110, 112, 115, 113, 118, 120]
    lower_ci = [105, 107, 110, 108, 113, 115]
    upper_ci = [115, 117, 120, 118, 123, 125]
    
    html = create_simple_forecast_chart(
        hist_data, 
        forecast_values, 
        future_dates,
        (lower_ci, upper_ci)
    )
    
    print("Simple chart created successfully!")
    print(f"HTML length: {len(html)}")
