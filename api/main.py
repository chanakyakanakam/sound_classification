"""
Path: /home/chanakya/sound_classification/api/main.py

FastAPI Backend for Pump-Net
RESTful API for audio anomaly detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
from pathlib import Path
import uvicorn
from config.config import settings
from utils.logger import logger
from src.inference import PumpNetInference

# Initialize FastAPI app
app = FastAPI(
    title="Pump-Net API",
    description="Industrial Pump Anomaly Detection API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference pipeline
try:
    inference_pipeline = PumpNetInference()
    logger.info("Inference pipeline loaded successfully")
except Exception as e:
    logger.error(f"Failed to load inference pipeline: {e}")
    inference_pipeline = None

# Response models
class PredictionResponse(BaseModel):
    prediction: str
    prediction_class: int
    probability_abnormal: float
    probability_normal: float
    confidence: float
    is_confident: bool
    threshold_used: float
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Pump-Net API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if inference_pipeline else "unhealthy",
        model_loaded=inference_pipeline is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict anomaly from uploaded audio file
    """
    if inference_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type - Fixed: Handle None filename
    if not file.filename or not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Make prediction
        result = inference_pipeline.predict_from_file(tmp_path)
        
        # Clean up temp file
        tmp_path.unlink()
        
        # Add message
        if result['prediction'] == 'Abnormal':
            message = "⚠️ ANOMALY DETECTED! Pump requires inspection."
        else:
            message = "✅ Normal operation. No anomalies detected."
        
        result['message'] = message
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch prediction for multiple audio files
    
    Args:
        files: List of audio files
    
    Returns:
        List of predictions
    """
    if inference_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = Path(tmp_file.name)
            
            result = inference_pipeline.predict_from_file(tmp_path)
            result['filename'] = file.filename
            results.append(result)
            
            tmp_path.unlink()
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    logger.info(f"Starting Pump-Net API on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
