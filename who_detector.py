"""
WHO Guideline-based Anomaly Detector for AQI
Uses World Health Organization air quality guidelines
"""

class WHOAnomalyDetector:
    """WHO guideline-based anomaly detection for air quality"""
    
    def __init__(self):
        """
        Initialize with WHO air quality guidelines
        All values in µg/m³ except CO (mg/m³)
        """
        # WHO Guidelines (24-hour means for daily monitoring)
        self.thresholds = {
            'PM2.5': 15.0,      # µg/m³ (24-hour mean)
            'PM10': 45.0,       # µg/m³ (24-hour mean)
            'CO': 4.0,          # mg/m³ (8-hour average, using as daily threshold)
            'NO2': 25.0,        # µg/m³ (24-hour mean)
            'AQI': 100.0        # AQI threshold (>100 = Unhealthy for sensitive groups)
        }
        
        # AQI categories
        self.aqi_categories = {
            (0, 50): "Good",
            (51, 100): "Moderate",
            (101, 150): "Unhealthy for Sensitive Groups",
            (151, 200): "Unhealthy",
            (201, 300): "Very Unhealthy",
            (301, 500): "Hazardous"
        }
        
        self.feature_names = ['PM2.5', 'PM10', 'CO', 'NO2']
    
    def get_aqi_category(self, aqi):
        """Get AQI category name"""
        for (low, high), category in self.aqi_categories.items():
            if low <= aqi <= high:
                return category
        return "Hazardous" if aqi > 300 else "Unknown"
    
    def predict_with_alerts(self, X, show_alerts=True):
        """
        Predict anomalies based on WHO guidelines
        
        Args:
            X: DataFrame with pollutant data
            show_alerts: Whether to print alerts
            
        Returns:
            anomalies, violations_df, alert_details
        """
        import pandas as pd
        import numpy as np
        
        anomalies = []
        alert_details = []
        
        if show_alerts:
            print("\n" + "=" * 80)
            print("WHO GUIDELINE-BASED AIR QUALITY ALERTS")
            print("=" * 80)
        
        anomaly_count = 0
        
        for i, idx in enumerate(X.index):
            violations = []
            is_anomaly = False
            
            # Check each pollutant against WHO guidelines
            for feature in self.feature_names:
                if feature in X.columns:
                    value = X.loc[idx, feature]
                    threshold = self.thresholds[feature]
                    
                    if value > threshold:
                        is_anomaly = True
                        exceedance = ((value - threshold) / threshold) * 100
                        
                        violation = {
                            'pollutant': feature,
                            'value': value,
                            'threshold': threshold,
                            'exceedance_percent': exceedance,
                            'status': f'{feature} exceeds WHO guidelines'
                        }
                        violations.append(violation)
            
            # Check AQI
            if 'AQI' in X.columns:
                aqi_value = X.loc[idx, 'AQI']
                aqi_category = self.get_aqi_category(aqi_value)
                
                if aqi_value > self.thresholds['AQI']:
                    is_anomaly = True
                    exceedance = ((aqi_value - self.thresholds['AQI']) / self.thresholds['AQI']) * 100
                    
                    violation = {
                        'pollutant': 'AQI',
                        'value': aqi_value,
                        'threshold': self.thresholds['AQI'],
                        'exceedance_percent': exceedance,
                        'status': f'AQI exceeds WHO guidelines ({aqi_category})'
                    }
                    violations.append(violation)
            
            anomalies.append(1 if is_anomaly else 0)
            
            if is_anomaly:
                anomaly_count += 1
                alert_info = {
                    'index': idx,
                    'city': X.loc[idx, 'City'] if 'City' in X.columns else 'N/A',
                    'date': X.loc[idx, 'Date'] if 'Date' in X.columns else 'N/A',
                    'aqi': X.loc[idx, 'AQI'] if 'AQI' in X.columns else 0,
                    'aqi_category': aqi_category if 'AQI' in X.columns else 'N/A',
                    'violations': violations
                }
                alert_details.append(alert_info)
                
                if show_alerts:
                    print(f"\n[ALERT #{anomaly_count}] AIR QUALITY VIOLATION!")
                    if 'Date' in X.columns:
                        print(f"Date: {alert_info['date']}")
                    if 'City' in X.columns:
                        print(f"City: {alert_info['city']}")
                    if 'AQI' in X.columns:
                        print(f"AQI: {alert_info['aqi']:.2f} ({alert_info['aqi_category']})")
                    
                    violated_pollutants = [v['pollutant'] for v in violations]
                    print(f"Violations: {', '.join(violated_pollutants)}")
                    print("-" * 80)
                    
                    for violation in violations:
                        print(f"  {violation['pollutant']}:")
                        print(f"    Current Value: {violation['value']:.2f}")
                        print(f"    WHO Guideline: {violation['threshold']:.2f}")
                        print(f"    Exceedance: {violation['exceedance_percent']:.1f}% above guideline")
                        print(f"    Status: {violation['status']}")
        
        if show_alerts:
            print("\n" + "=" * 80)
            print(f"Total Violations Detected: {anomaly_count}")
            print("=" * 80)
        
        return np.array(anomalies), None, alert_details
    
    def predict(self, X):
        """Simple prediction without alerts"""
        anomalies, _, _ = self.predict_with_alerts(X, show_alerts=False)
        return anomalies, None
