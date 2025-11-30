"""
AQI Anomaly Detection Model using Z-Score Method
Analyzes air quality data to detect anomalies and identify key pollutants affecting AQI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class AQIAnomalyDetector:
    """Z-Score based anomaly detection for AQI data"""
    
    def __init__(self, threshold=3.0):
        """
        Initialize the anomaly detector
        
        Args:
            threshold: Z-score threshold for anomaly detection (default: 3.0)
        """
        self.threshold = threshold
        self.means = {}
        self.stds = {}
        self.feature_names = ['PM2.5', 'PM10', 'CO', 'NO2']
        
    def fit(self, X):
        """
        Calculate mean and standard deviation from training data
        
        Args:
            X: Training data (pandas DataFrame)
        """
        for feature in self.feature_names:
            if feature in X.columns:
                self.means[feature] = X[feature].mean()
                self.stds[feature] = X[feature].std()
        
        print("\n=== Model Training Statistics ===")
        for feature in self.feature_names:
            if feature in self.means:
                print(f"{feature}: Mean={self.means[feature]:.2f}, Std={self.stds[feature]:.2f}")
    
    def calculate_z_scores(self, X):
        """
        Calculate z-scores for each feature
        
        Args:
            X: Data to calculate z-scores for
            
        Returns:
            DataFrame with z-scores
        """
        z_scores = pd.DataFrame()
        
        for feature in self.feature_names:
            if feature in X.columns and feature in self.means:
                z_scores[f'{feature}_zscore'] = (X[feature] - self.means[feature]) / self.stds[feature]
        
        return z_scores
    
    def predict(self, X):
        """
        Predict anomalies based on z-scores
        
        Args:
            X: Data to predict anomalies for
            
        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        z_scores = self.calculate_z_scores(X)
        
        # An observation is anomalous if ANY feature has |z-score| > threshold
        anomalies = (np.abs(z_scores) > self.threshold).any(axis=1).astype(int)
        
        return anomalies.values, z_scores
    
    def predict_with_alerts(self, X, show_alerts=True):
        """
        Predict anomalies and generate real-time alerts for threshold violations
        
        Args:
            X: Data to predict anomalies for
            show_alerts: Whether to print alerts (default: True)
            
        Returns:
            Array of predictions, z_scores, and alert details
        """
        z_scores = self.calculate_z_scores(X)
        anomalies = (np.abs(z_scores) > self.threshold).any(axis=1).astype(int)
        
        alert_details = []
        
        if show_alerts:
            print("\n" + "=" * 80)
            print("REAL-TIME ANOMALY ALERTS")
            print("=" * 80)
            
            anomaly_count = 0
            for i, idx in enumerate(X.index):
                if anomalies.iloc[i] == 1:
                    anomaly_count += 1
                    alert_info = {
                        'index': idx,
                        'city': X.loc[idx, 'City'] if 'City' in X.columns else 'N/A',
                        'date': X.loc[idx, 'Date'] if 'Date' in X.columns else 'N/A',
                        'aqi': X.loc[idx, 'AQI'],
                        'violations': []
                    }
                    
                    # Check which pollutants exceeded threshold
                    violated_features = []
                    for feature in self.feature_names:
                        zscore_col = f'{feature}_zscore'
                        if zscore_col in z_scores.columns:
                            z_val = z_scores.iloc[i][zscore_col]
                            if abs(z_val) > self.threshold:
                                actual_value = X.loc[idx, feature]
                                normal_range_min = self.means[feature] - self.threshold * self.stds[feature]
                                normal_range_max = self.means[feature] + self.threshold * self.stds[feature]
                                
                                violation = {
                                    'pollutant': feature,
                                    'value': actual_value,
                                    'z_score': z_val,
                                    'normal_range': (normal_range_min, normal_range_max),
                                    'mean': self.means[feature]
                                }
                                alert_info['violations'].append(violation)
                                violated_features.append(feature)
                    
                    alert_details.append(alert_info)
                    
                    # Print alert
                    print(f"\n[ALERT #{anomaly_count}] ANOMALY DETECTED!")
                    if 'Date' in X.columns:
                        print(f"Date: {alert_info['date']}")
                    if 'City' in X.columns:
                        print(f"City: {alert_info['city']}")
                    print(f"AQI: {alert_info['aqi']:.2f}")
                    print(f"Violated Pollutants: {', '.join(violated_features)}")
                    print("-" * 80)
                    
                    for violation in alert_info['violations']:
                        status = "HIGH" if violation['value'] > violation['mean'] else "LOW"
                        print(f"  {violation['pollutant']}:")
                        print(f"    Current Value: {violation['value']:.2f} [{status}]")
                        print(f"    Z-Score: {violation['z_score']:.2f}")
                        print(f"    Normal Range: {violation['normal_range'][0]:.2f} - {violation['normal_range'][1]:.2f}")
                        print(f"    Deviation: {abs(violation['value'] - violation['mean']):.2f} from mean ({violation['mean']:.2f})")
            
            print("\n" + "=" * 80)
            print(f"Total Anomalies Detected: {anomaly_count}")
            print("=" * 80)
        
        return anomalies.values, z_scores, alert_details


def load_and_clean_data(filepath):
    """
    Load and clean the AQI dataset
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Cleaned DataFrame
    """
    print("=== Loading Data ===")
    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Focus on key features
    key_features = ['City', 'Date', 'PM2.5', 'PM10', 'CO', 'NO2', 'AQI']
    df = df[key_features]
    
    print(f"\n=== Data Quality Before Cleaning ===")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Remove rows where AQI is missing (we need this for labeling)
    df = df.dropna(subset=['AQI'])
    
    # Remove rows where ALL pollutant features are missing
    pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO2']
    df = df.dropna(subset=pollutant_cols, how='all')
    
    # Fill remaining missing values with column median
    for col in pollutant_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove outliers using IQR method for initial cleaning
    for col in pollutant_cols + ['AQI']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive outlier removal
        upper_bound = Q3 + 3 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"\n=== Data Quality After Cleaning ===")
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nData statistics:\n{df[pollutant_cols + ['AQI']].describe()}")
    
    return df


def analyze_correlations(df):
    """
    Analyze correlations between pollutants and AQI
    
    Args:
        df: DataFrame with pollutant and AQI data
        
    Returns:
        Correlation results
    """
    print("\n=== Correlation Analysis ===")
    
    pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO2']
    
    # Calculate correlations with AQI
    correlations = {}
    for col in pollutant_cols:
        corr = df[col].corr(df['AQI'])
        correlations[col] = corr
        print(f"{col} vs AQI: {corr:.4f}")
    
    # Find most responsible pollutant
    most_responsible = max(correlations, key=correlations.get)
    print(f"\n*** Most Responsible Pollutant for AQI: {most_responsible} (correlation: {correlations[most_responsible]:.4f}) ***")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[pollutant_cols + ['AQI']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Pollutants vs AQI', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nSaved: correlation_heatmap.png")
    
    # Create scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(pollutant_cols):
        axes[idx].scatter(df[col], df['AQI'], alpha=0.5, s=20)
        axes[idx].set_xlabel(col, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('AQI', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{col} vs AQI (r={correlations[col]:.3f})', fontsize=14)
        
        # Add trend line
        z = np.polyfit(df[col], df['AQI'], 1)
        p = np.poly1d(z)
        axes[idx].plot(df[col], p(df[col]), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('pollutant_aqi_scatter.png', dpi=300, bbox_inches='tight')
    print("Saved: pollutant_aqi_scatter.png")
    
    return correlations, most_responsible


def create_labels(df, threshold_percentile=90):
    """
    Create anomaly labels based on AQI threshold
    
    Args:
        df: DataFrame with AQI data
        threshold_percentile: Percentile to use as threshold for anomalies
        
    Returns:
        Array of labels (1 for anomaly, 0 for normal)
    """
    # Use the threshold_percentile of AQI as the anomaly threshold
    threshold = df['AQI'].quantile(threshold_percentile / 100)
    labels = (df['AQI'] > threshold).astype(int)
    
    print(f"\n=== Anomaly Labeling ===")
    print(f"AQI threshold (at {threshold_percentile}th percentile): {threshold:.2f}")
    print(f"Normal samples: {(labels == 0).sum()} ({(labels == 0).sum() / len(labels) * 100:.1f}%)")
    print(f"Anomaly samples: {(labels == 1).sum()} ({(labels == 1).sum() / len(labels) * 100:.1f}%)")
    
    return labels.values, threshold


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n=== Model Performance Metrics ===")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr * 100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nSaved: confusion_matrix.png")
    
    return accuracy, precision, recall, f1, fpr


def visualize_results(X_test, y_test, y_pred, z_scores):
    """
    Visualize anomaly detection results
    
    Args:
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels
        z_scores: Z-scores for test data
    """
    # Plot z-score distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    feature_names = ['PM2.5', 'PM10', 'CO', 'NO2']
    
    for idx, feature in enumerate(feature_names):
        zscore_col = f'{feature}_zscore'
        if zscore_col in z_scores.columns:
            axes[idx].hist(z_scores[zscore_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].axvline(x=3, color='r', linestyle='--', linewidth=2, label='Threshold (+3)')
            axes[idx].axvline(x=-3, color='r', linestyle='--', linewidth=2, label='Threshold (-3)')
            axes[idx].set_xlabel('Z-Score', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{feature} Z-Score Distribution', fontsize=14)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('z_score_distributions.png', dpi=300, bbox_inches='tight')
    print("\nSaved: z_score_distributions.png")
    
    # Plot anomaly detection results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(feature_names):
        if feature in X_test.columns:
            # Plot normal vs anomaly points
            normal_mask = y_pred == 0
            anomaly_mask = y_pred == 1
            
            axes[idx].scatter(X_test[feature][normal_mask], X_test['AQI'][normal_mask], 
                            alpha=0.5, s=30, c='green', label='Normal', edgecolors='black', linewidth=0.5)
            axes[idx].scatter(X_test[feature][anomaly_mask], X_test['AQI'][anomaly_mask], 
                            alpha=0.7, s=50, c='red', label='Anomaly', edgecolors='black', linewidth=0.5)
            
            axes[idx].set_xlabel(feature, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('AQI', fontsize=12, fontweight='bold')
            axes[idx].set_title(f'Anomaly Detection: {feature} vs AQI', fontsize=14)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    print("Saved: anomaly_detection_results.png")


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("AQI ANOMALY DETECTION MODEL - Z-SCORE METHOD")
    print("=" * 80)
    
    # Load and clean data
    df = load_and_clean_data('city_day.csv')
    
    # Analyze correlations
    correlations, most_responsible = analyze_correlations(df)
    
    # Create labels based on AQI threshold (90th percentile)
    labels, aqi_threshold = create_labels(df, threshold_percentile=90)
    
    # Prepare features (keep City and Date for alerts)
    feature_cols = ['City', 'Date', 'PM2.5', 'PM10', 'CO', 'NO2', 'AQI']
    X = df[feature_cols].copy()
    y = labels
    
    # Split data: 80% train, 20% test
    print("\n=== Train-Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")
    
    # Initialize and train model
    print("\n=== Training Anomaly Detection Model ===")
    detector = AQIAnomalyDetector(threshold=3.0)
    detector.fit(X_train)
    
    # Make predictions on test set
    print("\n=== Testing Model ===")
    y_pred, z_scores = detector.predict(X_test)
    
    # Evaluate model
    accuracy, precision, recall, f1, fpr = evaluate_model(y_test, y_pred)
    
    # Check if accuracy is around 90%
    if 0.85 <= accuracy <= 0.95:
        print("\n[OK] Target accuracy achieved (~90%)")
    else:
        print(f"\n[WARNING] Accuracy is {accuracy * 100:.2f}%, adjusting threshold might help")
    
    # Check false positive rate
    if fpr < 0.10:
        print("[OK] False positive rate is acceptable (<10%)")
    else:
        print(f"[WARNING] False positive rate is {fpr * 100:.2f}%, consider adjusting threshold")
    
    # Visualize results
    print("\n=== Generating Visualizations ===")
    # Remove City and Date for visualization
    X_test_viz = X_test.drop(columns=['City', 'Date'])
    visualize_results(X_test_viz, y_test, y_pred, z_scores)
    
    # Generate real-time alerts for detected anomalies
    print("\n=== Generating Real-Time Alerts for Anomalies ===")
    # Only show first 10 alerts to avoid overwhelming output
    anomaly_indices = X_test[y_pred == 1].head(10).index
    X_test_sample = X_test.loc[anomaly_indices]
    
    if len(X_test_sample) > 0:
        print(f"\nShowing alerts for first {len(X_test_sample)} detected anomalies...")
        _, _, alert_details = detector.predict_with_alerts(X_test_sample, show_alerts=True)
        
        # Save alert details to file
        alert_log = []
        for alert in alert_details:
            log_entry = f"\nDate: {alert['date']}, City: {alert['city']}, AQI: {alert['aqi']:.2f}\n"
            for v in alert['violations']:
                log_entry += f"  - {v['pollutant']}: {v['value']:.2f} (z-score: {v['z_score']:.2f})\n"
            alert_log.append(log_entry)
        
        with open('anomaly_alerts.log', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ANOMALY ALERT LOG\n")
            f.write("=" * 80 + "\n")
            f.writelines(alert_log)
        
        print("\n[INFO] Full alert log saved to: anomaly_alerts.log")
    else:
        print("\n[INFO] No anomalies detected in the test set.")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Most Responsible Pollutant for AQI: {most_responsible}")
    print(f"Correlation with AQI: {correlations[most_responsible]:.4f}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"False Positive Rate: {fpr * 100:.2f}%")
    print(f"Total Anomalies Detected: {(y_pred == 1).sum()}")
    print("\nGenerated Files:")
    print("  - correlation_heatmap.png")
    print("  - pollutant_aqi_scatter.png")
    print("  - confusion_matrix.png")
    print("  - z_score_distributions.png")
    print("  - anomaly_detection_results.png")
    print("  - anomaly_alerts.log")
    print("=" * 80)


if __name__ == "__main__":
    main()
