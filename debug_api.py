import requests
import json

def test_forecast_api():
    """Test the forecast API directly"""
    try:
        print("Testing OCC Forecast API...")
        
        # Test POST request
        data = {
            'n_months': 6,
            'model_name': 'prophet'
        }
        
        response = requests.post('http://127.0.0.1:5000/forecast', 
                               data=data, 
                               timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                json_data = response.json()
                print("✅ SUCCESS - JSON Response received:")
                print(f"  Success: {json_data.get('success')}")
                print(f"  Model Used: {json_data.get('model_used')}")
                print(f"  Forecast Length: {len(json_data.get('forecast', []))}")
                print(f"  Chart HTML: {'chart_html' in json_data}")
                print(f"  Future Dates: {len(json_data.get('future_dates', []))}")
                
                if json_data.get('success'):
                    print("\n🎉 API is working correctly!")
                    return True
                else:
                    print(f"\n❌ API returned error: {json_data.get('error')}")
                    return False
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"Raw response: {response.text[:500]}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    test_forecast_api()
