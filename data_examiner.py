import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def examine_excel_data():
    """Examine the Excel data structure and content"""
    try:
        # Read the Excel file
        df = pd.read_excel('US_OCC_PRICES_Sep25.xlsx')
        
        print("=== EXCEL DATA EXAMINATION ===")
        print(f"File shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head(10))
        print("\nLast few rows:")
        print(df.tail(10))
        print("\nData types:")
        print(df.dtypes)
        print("\nBasic statistics:")
        print(df.describe())
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Check date format and range
        if 'Month' in df.columns:
            print(f"\nDate range: {df['Month'].min()} to {df['Month'].max()}")
        
        # Save a sample to CSV for inspection
        df.to_csv('data_sample.csv', index=False)
        print("\nData saved to 'data_sample.csv' for inspection")
        
        return df
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

if __name__ == "__main__":
    data = examine_excel_data()
