"""
Path: /home/chanakya/sound_classification/api/main.py

FastAPI Backend for Pump-Net
RESTful API for audio anomaly detection
"""
import os
import sys
import contextlib

# Suppress TensorFlow logs BEFORE import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Context manager to suppress stderr (for absl warnings)
@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output"""
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

# Suppress all Python warnings
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow specific warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Import TensorFlow with stderr suppressed (prevents absl C++ warnings)
with suppress_stderr():
    import tensorflow as tf

from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from pathlib import Path
import uvicorn
import numpy as np

# Silent imports
from config.config import settings
from utils.logger import logger
from src.inference import PumpNetInference

# Global variables (initialized later in startup event)
inference_pipeline = None
explainability_available = False

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

# Startup event - runs ONCE when server starts
@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup"""
    global inference_pipeline, explainability_available
    
    try:
        logger.info("Initializing inference pipeline...")
        
        # Suppress stderr during model loading (prevents GPU messages)
        with suppress_stderr():
            inference_pipeline = PumpNetInference()
        
        # Check explainability with defensive coding
        if inference_pipeline.explainability_engine is not None:
            engine = inference_pipeline.explainability_engine
            
            # Double-check training_data exists
            if engine.training_data is not None and isinstance(engine.training_data, dict):
                try:
                    # Safely access training data
                    features = engine.training_data.get('features')
                    labels = engine.training_data.get('labels')
                    
                    if features is not None and labels is not None:
                        explainability_available = True
                        n_samples = len(features)
                        n_normal = int(np.sum(labels == 0))
                        n_abnormal = int(np.sum(labels == 1))
                        
                        logger.info("=" * 70)
                        logger.info("✅ PUMP-NET API READY")
                        logger.info("=" * 70)
                        logger.info(f"🤖 Model:         Loaded")
                        logger.info(f"🧠 Explainability: Enabled")
                        logger.info(f"📊 Training Data:  {n_samples} samples ({n_normal} normal, {n_abnormal} abnormal)")
                        logger.info(f"📍 API:           http://{settings.API_HOST}:{settings.API_PORT}")
                        logger.info(f"📚 Docs:          http://{settings.API_HOST}:{settings.API_PORT}/docs")
                        logger.info("=" * 70)
                    else:
                        logger.warning("⚠️ Explainability: training data features/labels are None")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not read training data details: {e}")
            else:
                logger.warning("⚠️ Explainability engine loaded but training data not available")
        else:
            logger.warning("⚠️ Explainability engine not available")
        
    except Exception as e:
        logger.error(f"❌ Failed to load inference pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
    explainability: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explainability_available: bool
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
    exp_available = False
    
    # Safely check explainability
    if inference_pipeline and inference_pipeline.explainability_engine:
        engine = inference_pipeline.explainability_engine
        if engine.training_data is not None and isinstance(engine.training_data, dict):
            exp_available = engine.training_data.get('features') is not None
    
    return HealthResponse(
        status="healthy" if inference_pipeline else "unhealthy",
        model_loaded=inference_pipeline is not None,
        explainability_available=exp_available,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict anomaly from uploaded audio file"""
    if inference_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename or not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        logger.info(f"Processing: {file.filename}")
        
        result = inference_pipeline.predict_from_file(tmp_path, include_explainability=True)
        
        tmp_path.unlink()
        
        if result['prediction'] == 'Abnormal':
            message = "⚠️ ANOMALY DETECTED! Pump requires inspection."
        else:
            message = "✅ Normal operation. No anomalies detected."
        
        result['message'] = message
        
        logger.info(f"Result: {result['prediction']} ({result['confidence']*100:.1f}%)")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple audio files"""
    if inference_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = Path(tmp_file.name)
            
            result = inference_pipeline.predict_from_file(tmp_path, include_explainability=False)
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
    print("\n" + "="*70)
    print("🚀 STARTING PUMP-NET API")
    print("="*70)
    print(f"📍 URL:  http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📚 Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("="*70)
    print("⏳ Loading models... Please wait...")
    print("="*70 + "\n")
    
    # Use server directly to avoid worker spawn issues
    config = uvicorn.Config(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    server.run()