#!/usr/bin/env python3
"""
Quick test to verify model fixes
"""
import requests
import json

# Test all models
models_to_test = ['arima', 'lstm', 'prophet', 'exponential_smoothing', 'random_forest']

print("Testing all forecasting models...")
print("=" * 50)

for model in models_to_test:
    print(f"\n🔍 Testing {model.upper()} model...")
    
    try:
        response = requests.post(
            'http://127.0.0.1:5000/forecast',
            data={'n_months': 6, 'model_name': model}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                forecast = data.get('forecast', [])
                model_used = data.get('model_used', 'unknown')
                print(f"✅ {model.upper()}: SUCCESS")
                print(f"   Model used: {model_used}")
                print(f"   Forecast length: {len(forecast)}")
                if len(forecast) >= 3:
                    print(f"   Sample forecasts: {forecast[:3]}")
                    # Check if forecasts are varying (not all the same)
                    if len(set([round(f, 2) for f in forecast[:3]])) > 1:
                        print(f"   ✅ Forecasts are varying (dynamic)")
                    else:
                        print(f"   ⚠️ Forecasts appear static")
            else:
                print(f"❌ {model.upper()}: API returned success=False")
                print(f"   Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ {model.upper()}: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ {model.upper()}: Exception - {e}")

print("\n" + "=" * 50)
print("Model testing complete!")
