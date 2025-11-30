"""
Quick test to verify WHO detector works
"""
import pandas as pd
from who_detector import WHOAnomalyDetector

# Create test data
test_data = {
    'PM2.5': [180.5],  # Way above 15 threshold
    'PM10': [250.0],   # Way above 45 threshold
    'CO': [1.2],       # Below 4 threshold
    'NO2': [45.0],     # Above 25 threshold
    'AQI': [350.0]     # Above 100 threshold
}

df = pd.DataFrame(test_data)

# Test detector
detector = WHOAnomalyDetector()
anomalies, _, alert_details = detector.predict_with_alerts(df, show_alerts=True)

print(f"\nIs Anomaly: {anomalies[0] == 1}")
print(f"Number of violations: {len(alert_details[0]['violations']) if alert_details else 0}")
