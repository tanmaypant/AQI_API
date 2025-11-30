# AQI Anomaly Detection API - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server
```bash
python api.py
```

The API will start on `http://localhost:8000`

### 3. Access Interactive Documentation
Open your browser and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API and model are running properly.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-30T15:15:38"
}
```

---

### 2. Model Information
**GET** `/model-info`

Get model statistics and thresholds.

**Example:**
```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "threshold": 3.0,
  "feature_means": {
    "PM2.5": 54.72,
    "PM10": 99.22,
    "CO": 0.95,
    "NO2": 24.92
  },
  "feature_stds": {
    "PM2.5": 38.80,
    "PM10": 51.18,
    "CO": 0.62,
    "NO2": 17.29
  },
  "features": ["PM2.5", "PM10", "CO", "NO2"],
  "model_loaded": true
}
```

---

### 3. Single Prediction
**POST** `/predict`

Predict if a single air quality measurement is anomalous.

**Request Body:**
```json
{
  "pm25": 180.5,
  "pm10": 250.0,
  "co": 1.2,
  "no2": 45.0,
  "aqi": 350.0,
  "city": "Delhi",
  "date": "2025-11-30"
}
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pm25": 180.5,
    "pm10": 250.0,
    "co": 1.2,
    "no2": 45.0,
    "aqi": 350.0,
    "city": "Delhi",
    "date": "2025-11-30"
  }'
```

**Example (Python):**
```python
import requests

data = {
    "pm25": 180.5,
    "pm10": 250.0,
    "co": 1.2,
    "no2": 45.0,
    "aqi": 350.0,
    "city": "Delhi",
    "date": "2025-11-30"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

**Response (Anomaly Detected):**
```json
{
  "is_anomaly": true,
  "aqi": 350.0,
  "city": "Delhi",
  "date": "2025-11-30",
  "violations": [
    {
      "pollutant": "PM2.5",
      "value": 180.5,
      "z_score": 3.24,
      "normal_range_min": -61.68,
      "normal_range_max": 171.12,
      "mean": 54.72,
      "status": "HIGH"
    }
  ],
  "message": "ANOMALY DETECTED!"
}
```

**Response (Normal):**
```json
{
  "is_anomaly": false,
  "aqi": 85.0,
  "city": "Mumbai",
  "date": "2025-11-30",
  "violations": [],
  "message": "Normal air quality"
}
```

---

### 4. Batch Prediction
**POST** `/predict-batch`

Predict anomalies for multiple measurements at once (max 1000).

**Request Body:**
```json
{
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
    }
  ]
}
```

**Example (Python):**
```python
import requests

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
        }
    ]
}

response = requests.post("http://localhost:8000/predict-batch", json=batch_data)
result = response.json()
print(f"Total: {result['total_samples']}, Anomalies: {result['anomalies_detected']}")
```

**Response:**
```json
{
  "total_samples": 2,
  "anomalies_detected": 1,
  "predictions": [
    {
      "is_anomaly": false,
      "aqi": 105.0,
      "city": "Mumbai",
      "date": "2025-11-30",
      "violations": [],
      "message": "Normal air quality"
    },
    {
      "is_anomaly": true,
      "aqi": 380.0,
      "city": "Delhi",
      "date": "2025-11-30",
      "violations": [
        {
          "pollutant": "PM2.5",
          "value": 200.0,
          "z_score": 3.74,
          "normal_range_min": -61.68,
          "normal_range_max": 171.12,
          "mean": 54.72,
          "status": "HIGH"
        },
        {
          "pollutant": "CO",
          "value": 3.5,
          "z_score": 4.13,
          "normal_range_min": -0.89,
          "normal_range_max": 2.80,
          "mean": 0.95,
          "status": "HIGH"
        }
      ],
      "message": "ANOMALY DETECTED!"
    }
  ]
}
```

---

## Complete Python Client Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Check health
health = requests.get(f"{BASE_URL}/health").json()
print("API Status:", health['status'])

# 2. Get model info
model_info = requests.get(f"{BASE_URL}/model-info").json()
print("Model Threshold:", model_info['threshold'])
print("Features:", model_info['features'])

# 3. Single prediction
data = {
    "pm25": 150.0,
    "pm10": 200.0,
    "co": 2.5,
    "no2": 60.0,
    "aqi": 280.0,
    "city": "Delhi",
    "date": "2025-11-30"
}

response = requests.post(f"{BASE_URL}/predict", json=data)
result = response.json()

if result['is_anomaly']:
    print(f"\n⚠️ ANOMALY DETECTED in {result['city']}!")
    print(f"AQI: {result['aqi']}")
    print("Violations:")
    for v in result['violations']:
        print(f"  - {v['pollutant']}: {v['value']:.2f} [{v['status']}]")
        print(f"    Z-Score: {v['z_score']:.2f}")
else:
    print(f"\n✓ Normal air quality in {result['city']}")
    print(f"AQI: {result['aqi']}")
```

---

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid input data)
- **503**: Service Unavailable (model not loaded)

**Example Error Response:**
```json
{
  "detail": "Model not loaded"
}
```

---

## Running in Production

For production deployment, use a production ASGI server:

```bash
# Using uvicorn with workers
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# Using gunicorn with uvicorn workers
gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Features

✓ **Real-time predictions** with detailed violation information
✓ **Batch processing** for multiple measurements
✓ **Interactive documentation** (Swagger UI)
✓ **CORS enabled** for web applications
✓ **Health checks** for monitoring
✓ **Model statistics** endpoint
✓ **Pydantic validation** for request/response
✓ **Detailed error messages**

---

## Notes

- The API loads the model on startup using the `city_day.csv` file
- All pollutant values must be non-negative
- Batch predictions are limited to 1000 samples per request
- City and date fields are optional
- Z-score threshold is set to 3.0 (configurable in code)
