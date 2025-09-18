import requests
import json

# Test the forecast endpoint
try:
    response = requests.post('http://127.0.0.1:5000/forecast',
                           data={'n_months': 6, 'model_name': 'prophet'},
                           timeout=30)
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print('✅ SUCCESS: Forecast generated!')
            print(f'Model: {result["model_used"]}')
            print(f'Forecast length: {len(result["forecast"])}')
            print(f'Chart included: {"chart_html" in result}')
            print(f'Sample forecast: {result["forecast"][:2]}')
        else:
            print(f'❌ API Error: {result.get("error")}')
    else:
        print(f'❌ HTTP Error: {response.status_code}')
        print(response.text)
except Exception as e:
    print(f'❌ Connection Error: {e}')
