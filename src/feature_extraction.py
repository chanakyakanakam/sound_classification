"""
Path: /home/chanakya/sound_classification/src/feature_extraction.py
Audio Feature Extraction Module
Extracts MFCCs, spectral, and temporal features from audio signals
"""

import numpy as np
import librosa
from typing import Tuple, Optional
from config.config import settings
from utils.logger import logger

class AudioFeatureExtractor:
    """Extract acoustic features from audio signals"""
    
    def __init__(self, n_mfcc: int = 13):
        self.n_mfcc = n_mfcc
        logger.info(f"AudioFeatureExtractor initialized with {n_mfcc} MFCCs")
    
    def extract_mfccs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCCs"""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            logger.debug(f"Extracted {len(mfccs_processed)} MFCCs")
            return mfccs_processed
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {e}")
            raise
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Tuple[float, float, float]:
        """Extract spectral features"""
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)[0]
            
            # Fix: Convert to Python float explicitly
            features = (
                float(np.mean(spectral_centroids)),
                float(np.mean(spectral_rolloff)),
                float(np.mean(spectral_contrast))
            )
            logger.debug(f"Extracted spectral features: {features}")
            return features
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            raise
    
    def extract_temporal_features(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract temporal features
        
        Returns:
            Tuple of (zero_crossing_rate, autocorrelation) as Python floats
        """
        try:
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            autocorrelation = librosa.autocorrelate(audio)
            
            # Fix: Convert numpy types to Python float explicitly
            features = (
                float(np.mean(zero_crossing_rate)),
                float(np.mean(autocorrelation))
            )
            logger.debug(f"Extracted temporal features: {features}")
            return features
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            raise
    
    def extract_all_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract all features from audio"""
        logger.info("Extracting all features from audio signal")
        features = []
        
        # MFCCs
        if settings.EXTRACT_MFCCS:
            mfccs = self.extract_mfccs(audio, sr)
            features.append(mfccs)
        
        # Spectral features
        if settings.EXTRACT_SPECTRAL:
            spectral = self.extract_spectral_features(audio, sr)
            features.extend(spectral)
        
        # Temporal features
        if settings.EXTRACT_TEMPORAL:
            temporal = self.extract_temporal_features(audio)
            features.extend(temporal)
        
        feature_vector = np.concatenate([np.atleast_1d(f) for f in features])
        logger.info(f"Extracted {len(feature_vector)} features total")
        
        return feature_vector
    
    @staticmethod
    def get_feature_names() -> list:
        """Get names of all extracted features"""
        names = []
        
        if settings.EXTRACT_MFCCS:
            names.extend([f'MFCC_{i+1}' for i in range(settings.N_MFCC)])
        
        if settings.EXTRACT_SPECTRAL:
            names.extend(['Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Contrast'])
        
        if settings.EXTRACT_TEMPORAL:
            names.extend(['Zero_Crossing_Rate', 'Autocorrelation'])
        
        return names
    