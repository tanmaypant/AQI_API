# Quick Start Guide - WHO Guideline-Based API

## What Changed
✓ Replaced z-score method with **WHO air quality guidelines**
✓ Now uses real health standards instead of statistical thresholds

## WHO Thresholds Now Used:
- **PM2.5**: 15 µg/m³ (24-hour mean)
- **PM10**: 45 µg/m³ (24-hour mean)
- **CO**: 4 mg/m³ (8-hour average)
- **NO2**: 25 µg/m³ (24-hour mean)
- **AQI**: 100 (Unhealthy for sensitive groups)

## To Start the API:
```bash
python api.py
```

## Test Example:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"pm25\": 180.5, \"pm10\": 250.0, \"co\": 1.2, \"no2\": 45.0, \"aqi\": 350.0}"
```

## Response Format (NEW):
```json
{
  "is_anomaly": true,
  "violations": [
    {
      "pollutant": "PM2.5",
      "value": 180.5,
      "threshold": 15.0,
      "exceedance_percent": 1103.3,
      "status": "EXCEEDS WHO GUIDELINE"
    }
  ]
}
```

**Files Created:**
- `who_detector.py` - New WHO-based detector
- `api.py` - Updated to use WHO guidelines

**Just run:** `python api.py` and you're ready!
