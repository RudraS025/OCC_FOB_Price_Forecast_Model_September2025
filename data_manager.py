"""
Data Management Module for OCC Price Forecasting
Handles data loading, preprocessing, updating, and validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import logging
from typing import Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages all data operations for the OCC price forecasting system"""
    
    def __init__(self, excel_file: str = "US_OCC_PRICES_Sep25.xlsx", 
                 data_file: str = "occ_price_data.csv"):
        """
        Initialize DataManager
        
        Args:
            excel_file: Path to the original Excel file
            data_file: Path to the CSV file for storing updated data
        """
        self.excel_file = excel_file
        self.data_file = data_file
        self.data = None
        self.metadata = {}
        
        # Load or create data
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize data from Excel file or existing CSV"""
        try:
            if os.path.exists(self.data_file):
                logger.info(f"Loading existing data from {self.data_file}")
                self.data = pd.read_csv(self.data_file, parse_dates=['Month'])
            else:
                logger.info(f"Loading initial data from {self.excel_file}")
                self.data = pd.read_excel(self.excel_file)
                self.data['Month'] = pd.to_datetime(self.data['Month'])
                # Save to CSV for future use
                self.data.to_csv(self.data_file, index=False)
            
            # Update metadata
            self._update_metadata()
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error initializing data: {e}")
            raise
    
    def _update_metadata(self):
        """Update metadata about the dataset"""
        if self.data is not None:
            self.metadata = {
                'total_records': len(self.data),
                'date_range_start': self.data['Month'].min().strftime('%Y-%m-%d'),
                'date_range_end': self.data['Month'].max().strftime('%Y-%m-%d'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'min_price': float(self.data['Price(USD/ton)'].min()),
                'max_price': float(self.data['Price(USD/ton)'].max()),
                'mean_price': float(self.data['Price(USD/ton)'].mean()),
                'std_price': float(self.data['Price(USD/ton)'].std())
            }
    
    def get_data(self) -> pd.DataFrame:
        """Get the current dataset"""
        return self.data.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return self.metadata.copy()
    
    def add_actual_value(self, month: str, price: float) -> bool:
        """
        Add or update an actual price value for a specific month
        
        Args:
            month: Month in 'YYYY-MM-DD' format or 'YYYY-MM' format
            price: Price value in USD/ton
            
        Returns:
            bool: Success status
        """
        try:
            # Parse the month
            if len(month) == 7:  # Format: YYYY-MM
                month = f"{month}-01"
            
            month_date = pd.to_datetime(month)
            
            # Check if month already exists
            mask = self.data['Month'] == month_date
            
            if mask.any():
                # Update existing record
                self.data.loc[mask, 'Price(USD/ton)'] = price
                logger.info(f"Updated price for {month_date.strftime('%Y-%m')} to ${price}")
            else:
                # Add new record
                new_row = pd.DataFrame({
                    'Month': [month_date],
                    'Price(USD/ton)': [price]
                })
                self.data = pd.concat([self.data, new_row], ignore_index=True)
                self.data = self.data.sort_values('Month').reset_index(drop=True)
                logger.info(f"Added new price record for {month_date.strftime('%Y-%m')}: ${price}")
            
            # Save updated data
            self.data.to_csv(self.data_file, index=False)
            self._update_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding actual value: {e}")
            return False
    
    def get_training_data(self, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get training data up to a specific date
        
        Args:
            end_date: End date for training data (format: 'YYYY-MM-DD')
                     If None, uses all available data
            
        Returns:
            pd.DataFrame: Training data
        """
        if end_date is None:
            return self.data.copy()
        
        end_date = pd.to_datetime(end_date)
        mask = self.data['Month'] <= end_date
        return self.data[mask].copy()
    
    def get_next_forecast_date(self) -> datetime:
        """Get the next month to forecast"""
        last_date = self.data['Month'].max()
        next_date = last_date + pd.DateOffset(months=1)
        return next_date
    
    def generate_future_dates(self, n_periods: int) -> list:
        """
        Generate future dates for forecasting
        
        Args:
            n_periods: Number of future periods to generate
            
        Returns:
            list: List of future dates
        """
        last_date = self.data['Month'].max()
        future_dates = []
        
        for i in range(1, n_periods + 1):
            future_date = last_date + pd.DateOffset(months=i)
            future_dates.append(future_date)
        
        return future_dates
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the dataset and return validation results
        
        Returns:
            dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing values found: {missing_values.to_dict()}")
        
        # Check for duplicate dates
        duplicate_dates = self.data['Month'].duplicated().sum()
        if duplicate_dates > 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Duplicate dates found: {duplicate_dates}")
        
        # Check for negative prices
        negative_prices = (self.data['Price(USD/ton)'] < 0).sum()
        if negative_prices > 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Negative prices found: {negative_prices}")
        
        # Check for price outliers (prices > 3 standard deviations from mean)
        mean_price = self.data['Price(USD/ton)'].mean()
        std_price = self.data['Price(USD/ton)'].std()
        outliers = ((self.data['Price(USD/ton)'] - mean_price).abs() > 3 * std_price).sum()
        if outliers > 0:
            validation_results['warnings'].append(f"Potential outliers found: {outliers}")
        
        # Check for chronological order
        if not self.data['Month'].is_monotonic_increasing:
            validation_results['warnings'].append("Data is not in chronological order")
        
        return validation_results
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the data"""
        if self.data is None:
            return {}
        
        summary = {
            'metadata': self.get_metadata(),
            'validation': self.validate_data(),
            'statistics': {
                'price_statistics': self.data['Price(USD/ton)'].describe().to_dict(),
                'recent_trend': self._analyze_recent_trend(),
                'seasonal_patterns': self._analyze_seasonality()
            }
        }
        
        return summary
    
    def _analyze_recent_trend(self, periods: int = 12) -> Dict[str, Any]:
        """Analyze recent price trend"""
        if len(self.data) < periods:
            periods = len(self.data)
        
        recent_data = self.data.tail(periods)
        
        # Calculate trend
        x = np.arange(len(recent_data))
        y = recent_data['Price(USD/ton)'].values
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        else:
            slope = 0
            trend_direction = 'insufficient_data'
        
        return {
            'trend_direction': trend_direction,
            'slope': float(slope),
            'recent_change': float(recent_data['Price(USD/ton)'].iloc[-1] - recent_data['Price(USD/ton)'].iloc[0]),
            'recent_volatility': float(recent_data['Price(USD/ton)'].std())
        }
    
    def _analyze_seasonality(self) -> Dict[str, Any]:
        """Basic seasonality analysis"""
        if len(self.data) < 12:
            return {'insufficient_data': True}
        
        # Add month column for analysis
        data_with_month = self.data.copy()
        data_with_month['month'] = data_with_month['Month'].dt.month
        
        # Calculate monthly averages
        monthly_avg = data_with_month.groupby('month')['Price(USD/ton)'].mean()
        
        return {
            'monthly_averages': monthly_avg.to_dict(),
            'highest_month': int(monthly_avg.idxmax()),
            'lowest_month': int(monthly_avg.idxmin()),
            'seasonal_range': float(monthly_avg.max() - monthly_avg.min())
        }
    
    def export_data(self, filename: str, format: str = 'csv') -> bool:
        """
        Export data to different formats
        
        Args:
            filename: Output filename
            format: Export format ('csv', 'excel', 'json')
            
        Returns:
            bool: Success status
        """
        try:
            if format.lower() == 'csv':
                self.data.to_csv(filename, index=False)
            elif format.lower() == 'excel':
                self.data.to_excel(filename, index=False)
            elif format.lower() == 'json':
                self.data.to_json(filename, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize data manager
    dm = DataManager()
    
    # Get data summary
    summary = dm.get_data_summary()
    print("Data Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Example: Add a new actual value
    # dm.add_actual_value("2025-10", 145.0)
    
    # Get training data
    training_data = dm.get_training_data()
    print(f"\nTraining data shape: {training_data.shape}")
    print(f"Date range: {training_data['Month'].min()} to {training_data['Month'].max()}")
