"""
Path: /home/chanakya/sound_classification/src/inference.py
Inference Module
Real-time prediction pipeline
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple
import tensorflow as tf
from config.config import settings
from utils.logger import logger
from src.feature_extraction import AudioFeatureExtractor
from src.data_preprocessing import AudioDataLoader
from typing import Optional
class PumpNetInference:
    """Inference pipeline for predictions"""
    
    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.scaler_path = scaler_path or settings.SCALER_PATH
        
        logger.info("Loading model and scaler...")
        self.model = tf.keras.models.load_model(str(self.model_path))     # type: ignore
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.feature_extractor = AudioFeatureExtractor(settings.N_MFCC)
        self.data_loader = AudioDataLoader(settings.SAMPLE_RATE)
        
        logger.info("PumpNetInference initialized successfully")
    
    def predict_from_file(self, audio_path: Path) -> Dict:
        """
        Predict from audio file
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio
        audio, sr = self.data_loader.load_audio_file(audio_path)
        
        # Predict
        return self.predict_from_audio(audio, sr)
    
    def predict_from_audio(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Predict from audio array
        
        Args:
            audio: Audio time series
            sample_rate: Sample rate
        
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.feature_extractor.extract_all_features(audio, sample_rate)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction_prob = self.model.predict(features_scaled, verbose=0)[0][0]
        prediction_class = 1 if prediction_prob > settings.PREDICTION_THRESHOLD else 0
        prediction_label = 'Abnormal' if prediction_class == 1 else 'Normal'
        
        # Confidence
        confidence = prediction_prob if prediction_class == 1 else 1 - prediction_prob
        is_confident = confidence > settings.CONFIDENCE_THRESHOLD
        
        result = {
            'prediction': prediction_label,
            'prediction_class': int(prediction_class),
            'probability_abnormal': float(prediction_prob),
            'probability_normal': float(1 - prediction_prob),
            'confidence': float(confidence),
            'is_confident': bool(is_confident),
            'threshold_used': float(settings.PREDICTION_THRESHOLD)
        }
        
        logger.info(f"Prediction: {prediction_label} (confidence: {confidence:.2%})")
        
        return result
