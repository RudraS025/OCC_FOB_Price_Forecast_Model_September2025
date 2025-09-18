#!/usr/bin/env python3
"""
Quick forecast test to verify the fix
"""

import requests
import json

def test_forecast_api():
    """Test the forecast API endpoint"""
    try:
        # Test data
        data = {
            'n_months': 6,
            'model_name': 'prophet'
        }
        
        # Make POST request to forecast endpoint
        response = requests.post('http://127.0.0.1:5000/forecast', data=data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Forecast API Test: SUCCESS")
                print(f"Model used: {result['model_used']}")
                print(f"Forecast values: {result['forecast'][:3]}...")
                print(f"Chart HTML generated: {'chart_html' in result}")
                return True
            else:
                print(f"❌ Forecast API Test: FAILED - {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Forecast API Test: FAILED - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Forecast API Test: FAILED - {e}")
        return False

if __name__ == "__main__":
    print("Testing OCC Price Forecasting Application...")
    print("=" * 50)
    
    success = test_forecast_api()
    
    if success:
        print("\n🎉 Application is working correctly!")
        print("You can now use the forecast feature at: http://127.0.0.1:5000/forecast")
    else:
        print("\n⚠️ Application has issues. Please check the logs.")
