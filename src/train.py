"""
Path: /home/chanakya/sound_classification/src/train.py
Training Pipeline Module
Complete training workflow for Pump-Net
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint       #type: ignore
from config.config import settings
from utils.logger import logger
from src.data_preprocessing import AudioDataLoader, AudioAugmentor
from src.feature_extraction import AudioFeatureExtractor
from src.model import PumpNetModel
from tqdm import tqdm

class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, normal_path: Path, abnormal_path: Path):
        self.normal_path = normal_path
        self.abnormal_path = abnormal_path
        self.data_loader = AudioDataLoader(settings.SAMPLE_RATE)
        self.augmentor = AudioAugmentor()
        self.feature_extractor = AudioFeatureExtractor(settings.N_MFCC)
        self.scaler = StandardScaler()
        
        logger.info("TrainingPipeline initialized")
    
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """
        Load and preprocess data
        
        Returns:
            Tuple of (normal_audio_list, abnormal_audio_list, sample_rate)
        """
        logger.info("="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        
        # Load normal data
        normal_audio, normal_labels, sr_normal = self.data_loader.load_directory(
            self.normal_path, label=0
        )
        
        # Load abnormal data
        abnormal_audio, abnormal_labels, sr_abnormal = self.data_loader.load_directory(
            self.abnormal_path, label=1
        )
        
        sample_rate = sr_normal if sr_normal else sr_abnormal
        
        logger.info(f"Loaded {len(normal_audio)} normal and {len(abnormal_audio)} abnormal samples")
        logger.info(f"Sample rate: {sample_rate} Hz")
        
        return normal_audio, abnormal_audio, sample_rate
    
    def extract_features(self, audio_list: List[np.ndarray], sample_rate: int, 
                        augment: bool = False) -> np.ndarray:
        """
        Extract features from audio list
        
        Args:
            audio_list: List of audio arrays
            sample_rate: Sample rate
            augment: Whether to apply augmentation
            
        Returns:
            Feature matrix
        """
        logger.info(f"Extracting features (augment={augment})")
        
        features = []
        for audio in tqdm(audio_list, desc="Feature extraction"):
            # Original
            feat = self.feature_extractor.extract_all_features(audio, sample_rate)
            features.append(feat)
            
            # Augmented
            if augment:
                aug_audio = self.augmentor.augment(audio, sample_rate)
                aug_feat = self.feature_extractor.extract_all_features(aug_audio, sample_rate)
                features.append(aug_feat)
        
        return np.array(features)
    
    def prepare_data(self, normal_audio: List[np.ndarray], 
                    abnormal_audio: List[np.ndarray], 
                    sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels
        
        Args:
            normal_audio: List of normal audio arrays
            abnormal_audio: List of abnormal audio arrays
            sample_rate: Sample rate
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info("="*80)
        logger.info("FEATURE EXTRACTION")
        logger.info("="*80)
        
        # Extract features
        normal_features = self.extract_features(normal_audio, sample_rate, augment=False)
        abnormal_features = self.extract_features(abnormal_audio, sample_rate, augment=True)
        
        # Combine
        X = np.concatenate((normal_features, abnormal_features))
        y = np.concatenate((
            np.zeros(normal_features.shape[0]),
            np.ones(abnormal_features.shape[0])
        ))
        
        logger.info(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}")
        logger.info(f"Normal: {np.sum(y==0)}, Abnormal: {np.sum(y==1)}")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train model
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info("="*80)
        logger.info("MODEL TRAINING")
        logger.info("="*80)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE, stratify=y, random_state=settings.RANDOM_STATE
        )
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        model_builder = PumpNetModel(X_train_scaled.shape[1], settings.LEARNING_RATE)
        model = model_builder.build()
        
        # Callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_auc', 
                patience=settings.PATIENCE, 
                restore_best_weights=True, 
                mode='max', 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=10, 
                min_lr=1e-7, 
                verbose=1
            ),
            ModelCheckpoint(
                str(settings.MODEL_PATH), 
                monitor='val_auc', 
                save_best_only=True, 
                mode='max', 
                verbose=1
            )
        ]
        
        # Train
        logger.info(f"Training for {settings.EPOCHS} epochs...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.15,
            epochs=settings.EPOCHS,
            batch_size=settings.BATCH_SIZE,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate
        logger.info("="*80)
        logger.info("EVALUATION")
        logger.info("="*80)
        
        y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, y_pred_prob))
        }
        
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        # Save scaler
        with open(settings.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {settings.SCALER_PATH}")
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        
        return metrics
    
    def run(self) -> Dict:
        """Run complete training pipeline"""
        normal_audio, abnormal_audio, sample_rate = self.load_data()
        X, y = self.prepare_data(normal_audio, abnormal_audio, sample_rate)
        metrics = self.train(X, y)
        return metrics


# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TrainingPipeline(
        normal_path=settings.NORMAL_PATH,
        abnormal_path=settings.ABNORMAL_PATH
    )
    
    # Run training
    results = pipeline.run()
    
    print("\nTraining Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")