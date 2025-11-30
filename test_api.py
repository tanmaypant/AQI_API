"""
Test client for AQI Anomaly Detection API
Demonstrates how to use the API endpoints
"""

import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*80)
    print("Testing Health Endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*80)
    print("Testing Model Info Endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Threshold: {result['threshold']}")
    print(f"Features: {result['features']}")
    print(f"Model Loaded: {result['model_loaded']}")

def test_single_prediction_normal():
    """Test single prediction with normal values"""
    print("\n" + "="*80)
    print("Testing Single Prediction - Normal Air Quality")
    print("="*80)
    
    data = {
        "pm25": 45.0,
        "pm10": 95.0,
        "co": 0.8,
        "no2": 20.0,
        "aqi": 105.0,
        "city": "Mumbai",
        "date": "2025-11-30"
    }
    
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    
    print(f"\nResult:")
    print(f"  Is Anomaly: {result['is_anomaly']}")
    print(f"  Message: {result['message']}")
    print(f"  AQI: {result['aqi']}")
    print(f"  City: {result['city']}")

def test_single_prediction_anomaly():
    """Test single prediction with anomalous values"""
    print("\n" + "="*80)
    print("Testing Single Prediction - Anomalous Air Quality")
    print("="*80)
    
    data = {
        "pm25": 200.0,
        "pm10": 280.0,
        "co": 3.5,
        "no2": 95.0,
        "aqi": 380.0,
        "city": "Delhi",
        "date": "2025-11-30"
    }
    
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    
    print(f"\nResult:")
    print(f"  Is Anomaly: {result['is_anomaly']}")
    print(f"  Message: {result['message']}")
    print(f"  AQI: {result['aqi']}")
    print(f"  City: {result['city']}")
    
    if result['violations']:
        print(f"\n  Violations:")
        for v in result['violations']:
            print(f"    - {v['pollutant']}:")
            print(f"        Value: {v['value']:.2f} [{v['status']}]")
            print(f"        Z-Score: {v['z_score']:.2f}")
            print(f"        Normal Range: {v['normal_range_min']:.2f} - {v['normal_range_max']:.2f}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*80)
    print("Testing Batch Prediction")
    print("="*80)
    
    batch_data = {
        "data": [
            {
                "pm25": 45.0,
                "pm10": 95.0,
                "co": 0.8,
                "no2": 20.0,
                "aqi": 105.0,
                "city": "Mumbai",
                "date": "2025-11-30"
            },
            {
                "pm25": 200.0,
                "pm10": 280.0,
                "co": 3.5,
                "no2": 95.0,
                "aqi": 380.0,
                "city": "Delhi",
                "date": "2025-11-30"
            },
            {
                "pm25": 30.0,
                "pm10": 70.0,
                "co": 0.5,
                "no2": 15.0,
                "aqi": 75.0,
                "city": "Bangalore",
                "date": "2025-11-30"
            }
        ]
    }
    
    print(f"Batch Size: {len(batch_data['data'])}")
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    
    print(f"\nResults:")
    print(f"  Total Samples: {result['total_samples']}")
    print(f"  Anomalies Detected: {result['anomalies_detected']}")
    
    print(f"\n  Individual Predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        status = "⚠️ ANOMALY" if pred['is_anomaly'] else "✓ Normal"
        print(f"    {i}. {pred['city']}: {status} (AQI: {pred['aqi']})")

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("AQI ANOMALY DETECTION API - TEST CLIENT")
    print("="*80)
    print("Make sure the API is running on http://localhost:8000")
    print("Start it with: python api.py")
    
    try:
        # Test all endpoints
        test_health()
        test_model_info()
        test_single_prediction_normal()
        test_single_prediction_anomaly()
        test_batch_prediction()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running: python api.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main()
