import requests
import time

# Wait a moment for the server
time.sleep(2)

print("Testing forecast API endpoint...")

try:
    # Test the actual endpoint that the JavaScript is calling
    response = requests.post('http://127.0.0.1:5000/forecast', 
                           data={'n_months': '3'},
                           headers={'Content-Type': 'application/x-www-form-urlencoded'})
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Raw Response Text: {response.text[:500]}...")  # First 500 chars
    
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"\nJSON Response Structure:")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  {key}: list of {len(value)} items -> {value[:3] if len(value) > 3 else value}")
                else:
                    print(f"  {key}: {type(value).__name__} -> {value}")
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
    else:
        print(f"Error response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("Connection refused - server might not be ready yet")
except Exception as e:
    print(f"Error: {e}")
