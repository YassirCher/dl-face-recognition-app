import requests
import json

url = 'http://127.0.0.1:5000/api/set_model'
data = {'backbone': 'efficientnet_b3', 'classifier': 'cosface'}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
