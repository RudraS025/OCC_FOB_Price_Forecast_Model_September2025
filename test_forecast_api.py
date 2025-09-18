import requests
import json

try:
    response = requests.post('http://127.0.0.1:5000/forecast', 
                           data={'n_months': 3})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        if data.get('success'):
            print(f"Forecast: {data.get('forecast')}")
            print(f"Future dates: {data.get('future_dates')}")
        else:
            print(f"Error: {data.get('error')}")
except Exception as e:
    print(f"Error: {e}")
