"""
Path: /home/chanakya/sound_classification/src/inference.py
Inference Module with Explainability Support
Real-time prediction pipeline with similarity-based explanations
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional
import tensorflow as tf
from config.config import settings
from utils.logger import logger
from src.feature_extraction import AudioFeatureExtractor
from src.data_preprocessing import AudioDataLoader
from src.explainability import ExplainabilityEngine


class PumpNetInference:
    """Inference pipeline for predictions with explainability"""
    
    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """
        Initialize inference pipeline
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.scaler_path = scaler_path or settings.SCALER_PATH

        logger.info("Loading model and scaler...")
        self.model = tf.keras.models.load_model(str(self.model_path))     # type: ignore
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.feature_extractor = AudioFeatureExtractor(settings.N_MFCC)
        self.data_loader = AudioDataLoader(settings.SAMPLE_RATE)
        
        # Initialize explainability engine
        try:
            self.explainability_engine = ExplainabilityEngine()
            logger.info("Explainability engine loaded successfully")
        except Exception as e:
            logger.warning(f"Explainability engine not available: {e}")
            self.explainability_engine = None
        
        logger.info("PumpNetInference initialized successfully")
    
    def predict_from_file(self, audio_path: Path, include_explainability: bool = True) -> Dict:
        """
        Predict from audio file
        
        Args:
            audio_path: Path to audio file
            include_explainability: Whether to include explainability data
            
        Returns:
            Dictionary with prediction and optional explainability data
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio
        audio, sr = self.data_loader.load_audio_file(audio_path)
        
        # Predict
        return self.predict_from_audio(audio, sr, include_explainability=include_explainability)
    
    def predict_from_audio(self, audio: np.ndarray, sample_rate: int, 
                          include_explainability: bool = True) -> Dict:
        """
        Predict from audio array
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            include_explainability: Whether to include explainability data
            
        Returns:
            Dictionary with prediction and optional explainability data
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
        
        # Add explainability if requested and available
        if include_explainability and self.explainability_engine is not None:
            try:
                explanation = self.explainability_engine.explain_prediction(features_scaled[0])
                result['explainability'] = explanation
                logger.info("Explainability data added to prediction")
            except Exception as e:
                logger.warning(f"Failed to generate explainability: {e}")
                result['explainability'] = {'available': False, 'error': str(e)}
        else:
            result['explainability'] = {'available': False}
        
        logger.info(f"Prediction: {prediction_label} (confidence: {confidence:.2%})")
        
        return result
    
    def predict_with_gradcam(self, audio_path: Path) -> Dict:
        """
        Predict with GradCAM visualization data
        (Kept for backward compatibility)
        """
        # Load audio
        audio, sr = self.data_loader.load_audio_file(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio, sr)
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
            'threshold_used': float(settings.PREDICTION_THRESHOLD),
            'audio': audio,
            'sample_rate': sr,
            'features': features_scaled,
            'model': self.model
        }
        
        return result
    
    