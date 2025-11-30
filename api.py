"""
FastAPI Application for AQI Anomaly Detection
Provides REST API endpoints for air quality anomaly detection
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Import the WHO detector class
import sys
sys.path.append(os.path.dirname(__file__))
from who_detector import WHOAnomalyDetector

# Initialize FastAPI app
app = FastAPI(
    title="AQI Anomaly Detection API",
    description="Real-time air quality anomaly detection using z-score method",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
detector = None
model_stats = {}

# Pydantic models for request/response
class PollutantData(BaseModel):
    """Single air quality measurement"""
    pm25: float = Field(..., description="PM2.5 concentration", ge=0)
    pm10: float = Field(..., description="PM10 concentration", ge=0)
    co: float = Field(..., description="CO concentration", ge=0)
    no2: float = Field(..., description="NO2 concentration", ge=0)
    aqi: float = Field(..., description="Air Quality Index", ge=0)
    city: Optional[str] = Field(None, description="City name")
    date: Optional[str] = Field(None, description="Date (YYYY-MM-DD)")

class BatchPollutantData(BaseModel):
    """Batch of air quality measurements"""
    data: List[PollutantData]

class ViolationDetail(BaseModel):
    """Details of a WHO guideline violation"""
    pollutant: str
    value: float
    threshold: float
    exceedance_percent: float
    status: str

class PredictionResponse(BaseModel):
    """Response for single prediction"""
    is_anomaly: bool
    aqi: float
    city: Optional[str]
    date: Optional[str]
    violations: List[ViolationDetail]
    message: str

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    total_samples: int
    anomalies_detected: int
    predictions: List[PredictionResponse]

class ModelInfo(BaseModel):
    """Model information and WHO guidelines"""
    method: str
    thresholds: Dict[str, float]
    features: List[str]
    model_loaded: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load the WHO guideline-based detector on startup"""
    global detector, model_stats
    
    try:
        # Initialize WHO-based detector
        detector = WHOAnomalyDetector()
        
        # Store model stats
        model_stats = {
            'method': 'WHO Guidelines',
            'thresholds': detector.thresholds,
            'features': detector.feature_names,
            'aqi_categories': detector.aqi_categories
        }
        
        print("[OK] WHO Guideline-based detector loaded successfully!")
        print("Thresholds:")
        for pollutant, threshold in detector.thresholds.items():
            unit = "mg/m³" if pollutant == "CO" else "µg/m³" if pollutant != "AQI" else "index"
            print(f"  {pollutant}: {threshold} {unit}")
        
    except Exception as e:
        print(f"[ERROR] Error loading detector: {e}")
        detector = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AQI Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "model_info": "/model-info",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if detector is not None else "unhealthy",
        "model_loaded": detector is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get WHO guideline thresholds and model information"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "method": "WHO Air Quality Guidelines",
        "thresholds": detector.thresholds,
        "features": detector.feature_names,
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_anomaly(data: PollutantData):
    """
    Predict if air quality measurement is anomalous
    
    Returns detailed information about any threshold violations
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create DataFrame from input
        input_data = {
            'PM2.5': [data.pm25],
            'PM10': [data.pm10],
            'CO': [data.co],
            'NO2': [data.no2],
            'AQI': [data.aqi]
        }
        
        if data.city:
            input_data['City'] = [data.city]
        if data.date:
            input_data['Date'] = [data.date]
        
        df = pd.DataFrame(input_data)
        
        # Make prediction
        predictions, z_scores, alert_details = detector.predict_with_alerts(df, show_alerts=False)
        
        is_anomaly = bool(predictions[0])
        violations = []
        
        if is_anomaly and alert_details:
            alert = alert_details[0]
            for v in alert['violations']:
                violations.append(ViolationDetail(
                    pollutant=v['pollutant'],
                    value=v['value'],
                    threshold=v['threshold'],
                    exceedance_percent=v['exceedance_percent'],
                    status=v['status']
                ))
        
        message = "ANOMALY DETECTED!" if is_anomaly else "Normal air quality"
        
        return {
            "is_anomaly": is_anomaly,
            "aqi": data.aqi,
            "city": data.city,
            "date": data.date,
            "violations": violations,
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchPollutantData):
    """
    Predict anomalies for multiple air quality measurements
    
    Processes a batch of measurements and returns predictions for each
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch.data) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 1000")
    
    try:
        predictions = []
        
        for item in batch.data:
            # Reuse the single prediction endpoint logic
            result = await predict_anomaly(item)
            predictions.append(result)
        
        anomaly_count = sum(1 for p in predictions if p.is_anomaly)
        
        return {
            "total_samples": len(predictions),
            "anomalies_detected": anomaly_count,
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting AQI Anomaly Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
