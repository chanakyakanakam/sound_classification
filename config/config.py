"""
Configuration Management for Pump-Net Production System
Path: /home/chanakya/sound_classification/config/config.py
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

class Settings:
    """Application settings and configuration"""
    
    def __init__(self):
        # ========================================================================
        # PROJECT PATHS (WSL2)
        # ========================================================================
        self.BASE_DIR = Path("/home/chanakya/sound_classification")
        
        # Original data locations (6_dB_pump/pump/id_00)
        self.ORIGINAL_DATA_DIR = self.BASE_DIR / "6_dB_pump" / "pump" / "id_00"
        self.ABNORMAL_PATH = self.ORIGINAL_DATA_DIR / "abnormal"
        self.NORMAL_PATH = self.ORIGINAL_DATA_DIR / "normal"
        
        # Model directory (no "data" subdirectory)
        self.MODELS_DIR = self.BASE_DIR / "models"
        
        # Visualization output
        self.VISUALIZATION_DIR = self.BASE_DIR / "visualizations"
        
        # ========================================================================
        # MODEL FILES
        # ========================================================================
        self.MODEL_PATH = self.MODELS_DIR / "pump_net_best_model.keras"
        self.SCALER_PATH = self.MODELS_DIR / "pump_net_scaler.pkl"
        self.TRAINING_REFERENCE_PATH = self.MODELS_DIR / "training_reference.pkl"
        
        # ========================================================================
        # AUDIO PARAMETERS
        # ========================================================================
        self.SAMPLE_RATE: Optional[int] = None  # Use native sample rate
        self.N_MFCC = 13
        self.MAX_AUDIO_LENGTH = 10
        
        # ========================================================================
        # FEATURE EXTRACTION FLAGS
        # ========================================================================
        self.EXTRACT_MFCCS = True
        self.EXTRACT_SPECTRAL = True
        self.EXTRACT_TEMPORAL = True
        
        # ========================================================================
        # DATA AUGMENTATION
        # ========================================================================
        self.AUGMENT_DATA = True
        self.NOISE_FACTOR = 0.005
        self.SHIFT_MAX = 0.2
        self.AUGMENTATION_FACTOR = 2
        
        # ========================================================================
        # DATA SPLIT RATIOS
        # ========================================================================
        self.TEST_SIZE = 0.2
        self.VAL_SIZE = 0.15
        self.RANDOM_STATE = 42
        
        # ========================================================================
        # MODEL ARCHITECTURE
        # ========================================================================
        self.HIDDEN_UNITS_1 = 64
        self.HIDDEN_UNITS_2 = 64
        self.DROPOUT_RATE = 0.5
        
        # ========================================================================
        # TRAINING PARAMETERS
        # ========================================================================
        self.BATCH_SIZE = 16
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.PATIENCE = 20
        
        # ========================================================================
        # CROSS-VALIDATION
        # ========================================================================
        self.USE_CROSS_VALIDATION = True
        self.CV_FOLDS = 5
        
        # ========================================================================
        # PREDICTION THRESHOLDS
        # ========================================================================
        self.PREDICTION_THRESHOLD = 0.5
        self.CONFIDENCE_THRESHOLD = 0.7
        
        # ========================================================================
        # EXPLAINABILITY SETTINGS
        # ========================================================================
        self.KNN_NEIGHBORS = 3  # Number of similar examples to find
        self.KNN_ADAPTIVE = False  # Use adaptive K based on dataset size
        self.PROJECTION_METHOD = "tsne"  # 'pca' or 'tsne'
        self.CLUSTERING_METHOD = "dbscan"  # 'dbscan' or 'kmeans'
        self.DBSCAN_EPS = 0.5
        self.DBSCAN_MIN_SAMPLES = 5
        
        # ========================================================================
        # API SETTINGS
        # ========================================================================
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000
        self.API_RELOAD = False  # Set to False for production (avoids triple initialization)
        self.API_WORKERS = 1
        self.CORS_ORIGINS = ["*"]
        
        # ========================================================================
        # STREAMLIT SETTINGS
        # ========================================================================
        self.STREAMLIT_HOST = "0.0.0.0"
        self.STREAMLIT_PORT = 8501
        self.STREAMLIT_THEME = "light"
        
        # ========================================================================
        # LOGGING CONFIGURATION
        # ========================================================================
        self.LOG_LEVEL = "INFO"
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        
        # ========================================================================
        # PERFORMANCE & OPTIMIZATION
        # ========================================================================
        self.USE_GPU = True
        self.GPU_MEMORY_GROWTH = True
        self.MIXED_PRECISION = False
        
        # ========================================================================
        # DATA LOADING
        # ========================================================================
        self.MAX_FILES_NORMAL: Optional[int] = None
        self.MAX_FILES_ABNORMAL: Optional[int] = None
        
        # Create directories
        self._create_directories()
        
        # Load from .env if exists
        self._load_env()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.MODELS_DIR,
            self.VISUALIZATION_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_env(self):
        """Load settings from .env file if it exists"""
        env_file = self.BASE_DIR / ".env"
        if env_file.exists():
            try:
                load_dotenv(env_file)
                
                # Override with environment variables if they exist
                self.API_HOST = os.getenv("API_HOST", self.API_HOST)
                self.API_PORT = int(os.getenv("API_PORT", self.API_PORT))
                self.STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", self.STREAMLIT_PORT))
                self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
                self.PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", self.PREDICTION_THRESHOLD))
                self.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", self.CONFIDENCE_THRESHOLD))
            except ImportError:
                pass  # python-dotenv not installed, use defaults
    
    def validate_data_paths(self) -> bool:
        """
        Validate that data paths exist
        
        Returns:
            True if all paths exist, False otherwise
        """
        required_paths = [
            (self.ORIGINAL_DATA_DIR, "Original data directory"),
            (self.ABNORMAL_PATH, "Abnormal data directory"),
            (self.NORMAL_PATH, "Normal data directory"),
        ]
        
        all_valid = True
        for path, description in required_paths:
            if not path.exists():
                print(f"WARNING: {description} not found: {path}")
                all_valid = False
            else:
                print(f"OK: {description} found: {path}")
        
        return all_valid
    
    def get_feature_count(self) -> int:
        """Calculate total number of features"""
        count = 0
        if self.EXTRACT_MFCCS:
            count += self.N_MFCC
        if self.EXTRACT_SPECTRAL:
            count += 3
        if self.EXTRACT_TEMPORAL:
            count += 2
        return count
    
    def summary(self) -> str:
        """Get configuration summary"""
        return f"""
{'='*80}
                     PUMP-NET CONFIGURATION SUMMARY
{'='*80}

DATA PATHS:
   Base Directory:    {self.BASE_DIR}
   Normal Data:       {self.NORMAL_PATH}
   Abnormal Data:     {self.ABNORMAL_PATH}
   Models Directory:  {self.MODELS_DIR}

AUDIO SETTINGS:
   Sample Rate:       {self.SAMPLE_RATE or 'Native'}
   MFCCs:             {self.N_MFCC}
   Max Length:        {self.MAX_AUDIO_LENGTH}s

FEATURES:
   Total Features:    {self.get_feature_count()}
   MFCCs:             {'Yes' if self.EXTRACT_MFCCS else 'No'}
   Spectral:          {'Yes' if self.EXTRACT_SPECTRAL else 'No'}
   Temporal:          {'Yes' if self.EXTRACT_TEMPORAL else 'No'}

MODEL ARCHITECTURE:
   Hidden Layer 1:    {self.HIDDEN_UNITS_1} units
   Hidden Layer 2:    {self.HIDDEN_UNITS_2} units
   Dropout Rate:      {self.DROPOUT_RATE}

TRAINING:
   Batch Size:        {self.BATCH_SIZE}
   Epochs:            {self.EPOCHS}
   Learning Rate:     {self.LEARNING_RATE}
   Cross-Validation:  {self.CV_FOLDS}-fold {'Yes' if self.USE_CROSS_VALIDATION else 'No'}

PREDICTION:
   Threshold:         {self.PREDICTION_THRESHOLD}
   Confidence:        {self.CONFIDENCE_THRESHOLD}

EXPLAINABILITY:
   KNN Neighbors:     {self.KNN_NEIGHBORS}
   Projection:        {self.PROJECTION_METHOD}
   Clustering:        {self.CLUSTERING_METHOD}

API SETTINGS:
   Host:              {self.API_HOST}
   Port:              {self.API_PORT}

STREAMLIT SETTINGS:
   Host:              {self.STREAMLIT_HOST}
   Port:              {self.STREAMLIT_PORT}

{'='*80}
"""


# Global settings instance
settings = Settings()


# Validate paths on import
if __name__ == "__main__":
    print(settings.summary())
    settings.validate_data_paths()