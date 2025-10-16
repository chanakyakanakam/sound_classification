"""
Path: /home/chanakya/sound_classification/src/train.py
Training Pipeline Module
Complete training workflow for Pump-Net with explainability support
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Any
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
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix

class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, normal_path: Path, abnormal_path: Path):
        self.normal_path = normal_path
        self.abnormal_path = abnormal_path
        self.data_loader = AudioDataLoader(settings.SAMPLE_RATE)
        self.augmentor = AudioAugmentor()
        self.feature_extractor = AudioFeatureExtractor(settings.N_MFCC)
        self.scaler = StandardScaler()
        
        # Store file paths for explainability
        self.file_paths = []
        
        logger.info("TrainingPipeline initialized")
    
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], int, List[str]]:
        """
        Load and preprocess data
        
        Returns:
            Tuple of (normal_audio_list, abnormal_audio_list, sample_rate, file_paths)
        """
        logger.info("="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        
        # Load normal data
        normal_audio, normal_labels, sr_normal = self.data_loader.load_directory(
            self.normal_path, label=0
        )
        normal_paths = [str(f) for f in list(self.normal_path.glob("*.wav"))[:len(normal_audio)]]
        
        # Load abnormal data
        abnormal_audio, abnormal_labels, sr_abnormal = self.data_loader.load_directory(
            self.abnormal_path, label=1
        )
        abnormal_paths = [str(f) for f in list(self.abnormal_path.glob("*.wav"))[:len(abnormal_audio)]]
        
        sample_rate = sr_normal if sr_normal else sr_abnormal
        
        # Combine file paths
        all_file_paths = normal_paths + abnormal_paths
        
        logger.info(f"Loaded {len(normal_audio)} normal and {len(abnormal_audio)} abnormal samples")
        logger.info(f"Sample rate: {sample_rate} Hz")
        
        return normal_audio, abnormal_audio, sample_rate, all_file_paths
    
    def extract_features(self, audio_list: List[np.ndarray], sample_rate: int, 
                        augment: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from audio list
        """
        logger.info(f"Extracting features (augment={augment})")
        
        features = []
        paths = []
        
        for i, audio in enumerate(tqdm(audio_list, desc="Feature extraction")):
            # Original
            feat = self.feature_extractor.extract_all_features(audio, sample_rate)
            features.append(feat)
            paths.append(None)  # Original samples - path tracked separately
            
            # Augmented
            if augment:
                aug_audio = self.augmentor.augment(audio, sample_rate)
                aug_feat = self.feature_extractor.extract_all_features(aug_audio, sample_rate)
                features.append(aug_feat)
                paths.append(None)  # Augmented samples don't have file paths
        
        return np.array(features), paths
    
    def prepare_data(self, normal_audio: List[np.ndarray], 
                    abnormal_audio: List[np.ndarray], 
                    sample_rate: int,
                    file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and labels
        """
        logger.info("="*80)
        logger.info("FEATURE EXTRACTION")
        logger.info("="*80)
        
        # Extract features
        normal_features, normal_paths = self.extract_features(normal_audio, sample_rate, augment=False)
        abnormal_features, abnormal_paths = self.extract_features(abnormal_audio, sample_rate, augment=True)
        
        # Update paths for original samples
        n_normal_orig = len(normal_audio)
        n_abnormal_orig = len(abnormal_audio)
        
        for i in range(n_normal_orig):
            normal_paths[i] = file_paths[i]
        
        for i in range(n_abnormal_orig):
            abnormal_paths[i] = file_paths[n_normal_orig + i]
        
        all_paths = normal_paths + abnormal_paths
        
        # Combine
        X = np.concatenate((normal_features, abnormal_features))
        y = np.concatenate((
            np.zeros(normal_features.shape[0]),
            np.ones(abnormal_features.shape[0])
        ))
        
        logger.info(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}")
        logger.info(f"Normal: {np.sum(y==0)}, Abnormal: {np.sum(y==1)}")
        
        return X, y, all_paths
    
    def save_training_reference(self, X_scaled: np.ndarray, y: np.ndarray, file_paths: List[str]):
        """
        Save training reference data for explainability
        """
        logger.info("="*80)
        logger.info("SAVING TRAINING REFERENCE FOR EXPLAINABILITY")
        logger.info("="*80)
        
        training_reference = {
            'features': X_scaled,
            'labels': y,
            'file_paths': file_paths,
            'feature_names': self.feature_extractor.get_feature_names(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }
        
        with open(settings.TRAINING_REFERENCE_PATH, 'wb') as f:
            pickle.dump(training_reference, f)
        
        logger.info(f"Training reference saved to {settings.TRAINING_REFERENCE_PATH}")
        logger.info(f"  - Samples: {len(y)}")
        logger.info(f"  - Features: {X_scaled.shape[1]}")
        logger.info(f"  - Normal: {np.sum(y==0)}")
        logger.info(f"  - Abnormal: {np.sum(y==1)}")

    def save_training_report(self, X_train: np.ndarray, X_val: np.ndarray, 
                            X_test: np.ndarray, y_train: np.ndarray, 
                            y_val: np.ndarray, y_test: np.ndarray,
                            metrics: Dict, history: Any) -> None:

        logger.info("="*80)
        logger.info("SAVING TRAINING REPORT")
        logger.info("="*80)
        
        # Get best epoch info
        best_epoch = int(np.argmax(history.history['val_auc'])) + 1
        total_epochs = len(history.history['loss'])
        

        y_pred_prob = metrics.get('y_pred_prob', None)  
        y_pred = metrics.get('y_pred', None)
        
        if y_pred is not None:
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        # Build comprehensive report
        report = {
            "metadata": {
                "model_name": "Pump-Net",
                "version": "1.0",
                "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "framework": "TensorFlow/Keras",
                "random_seed": settings.RANDOM_STATE
            },
            
            "dataset_info": {
                "total_samples": int(len(X_train) + len(X_val) + len(X_test)),
                "num_features": int(X_train.shape[1]),
                "feature_names": self.feature_extractor.get_feature_names(),
                
                "class_distribution": {
                    "total": {
                        "normal": int(np.sum(y_train == 0) + np.sum(y_val == 0) + np.sum(y_test == 0)),
                        "abnormal": int(np.sum(y_train == 1) + np.sum(y_val == 1) + np.sum(y_test == 1))
                    },
                    "ratio": f"1:{(np.sum(y_train == 0) + np.sum(y_val == 0) + np.sum(y_test == 0)) / (np.sum(y_train == 1) + np.sum(y_val == 1) + np.sum(y_test == 1)):.2f}"
                }
            },
            
            "data_split": {
                "strategy": "stratified_train_test_split",
                "test_size": float(settings.TEST_SIZE),
                "validation_size": float(settings.VAL_SIZE),
                
                "training": {
                    "total_samples": int(len(X_train)),
                    "percentage": float(len(X_train) / (len(X_train) + len(X_val) + len(X_test)) * 100),
                    "class_distribution": {
                        "normal": int(np.sum(y_train == 0)),
                        "abnormal": int(np.sum(y_train == 1))
                    },
                    "class_ratio": f"1:{np.sum(y_train == 0) / np.sum(y_train == 1):.2f}"
                },
                
                "validation": {
                    "total_samples": int(len(X_val)),
                    "percentage": float(len(X_val) / (len(X_train) + len(X_val) + len(X_test)) * 100),
                    "class_distribution": {
                        "normal": int(np.sum(y_val == 0)),
                        "abnormal": int(np.sum(y_val == 1))
                    },
                    "class_ratio": f"1:{np.sum(y_val == 0) / np.sum(y_val == 1):.2f}"
                },
                
                "testing": {
                    "total_samples": int(len(X_test)),
                    "percentage": float(len(X_test) / (len(X_train) + len(X_val) + len(X_test)) * 100),
                    "class_distribution": {
                        "normal": int(np.sum(y_test == 0)),
                        "abnormal": int(np.sum(y_test == 1))
                    },
                    "class_ratio": f"1:{np.sum(y_test == 0) / np.sum(y_test == 1):.2f}"
                }
            },
            
            "model_architecture": {
                "type": "Deep Neural Network",
                "layers": [
                    {"name": "input", "units": int(X_train.shape[1]), "activation": "none"},
                    {"name": "dense_1", "units": settings.HIDDEN_UNITS_1, "activation": "relu"},
                    {"name": "dropout_1", "rate": settings.DROPOUT_RATE},
                    {"name": "dense_2", "units": settings.HIDDEN_UNITS_2, "activation": "relu"},
                    {"name": "dropout_2", "rate": settings.DROPOUT_RATE},
                    {"name": "output", "units": 1, "activation": "sigmoid"}
                ],
                "total_parameters": 5441,
                "trainable_parameters": 5441
            },
            
            "training_configuration": {
                "optimizer": "Adam",
                "learning_rate": float(settings.LEARNING_RATE),
                "loss_function": "binary_crossentropy",
                "batch_size": int(settings.BATCH_SIZE),
                "max_epochs": int(settings.EPOCHS),
                "early_stopping_patience": int(settings.PATIENCE),
                "early_stopping_monitor": "val_auc"
            },
            
            "training_history": {
                "total_epochs_run": total_epochs,
                "best_epoch": best_epoch,
                "stopped_early": bool(total_epochs < settings.EPOCHS),
                
                "final_metrics": {
                    "train_accuracy": float(history.history['accuracy'][-1]),
                    "train_loss": float(history.history['loss'][-1]),
                    "val_accuracy": float(history.history['val_accuracy'][-1]),
                    "val_loss": float(history.history['val_loss'][-1]),
                    "val_auc": float(history.history['val_auc'][-1])
                },
                
                "best_epoch_metrics": {
                    "train_accuracy": float(history.history['accuracy'][best_epoch-1]),
                    "train_loss": float(history.history['loss'][best_epoch-1]),
                    "val_accuracy": float(history.history['val_accuracy'][best_epoch-1]),
                    "val_loss": float(history.history['val_loss'][best_epoch-1]),
                    "val_auc": float(history.history['val_auc'][best_epoch-1])
                }
            },
            
            "test_performance": {
                "metrics": {
                    "accuracy": {
                        "value": float(metrics['accuracy']),
                        "percentage": f"{metrics['accuracy']*100:.2f}%"
                    },
                    "precision": {
                        "value": float(metrics['precision']),
                        "percentage": f"{metrics['precision']*100:.2f}%"
                    },
                    "recall": {
                        "value": float(metrics['recall']),
                        "percentage": f"{metrics['recall']*100:.2f}%"
                    },
                    "f1_score": {
                        "value": float(metrics['f1_score']),
                        "percentage": f"{metrics['f1_score']*100:.2f}%"
                    },
                    "auc_roc": {
                        "value": float(metrics['auc_roc']),
                        "percentage": f"{metrics['auc_roc']*100:.2f}%"
                    },
                    "specificity": {
                        "value": float(metrics['specificity']),
                        "percentage": f"{metrics['specificity']*100:.2f}%",
                        "description": "True Negative Rate (correctly identified normal samples)"
                    },
                    "sensitivity": {
                        "value": float(metrics['sensitivity']),
                        "percentage": f"{metrics['sensitivity']*100:.2f}%",
                        "description": "True Positive Rate (correctly identified abnormal samples)"
                    }
                },
                
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                    "visualization": {
                        "matrix": [
                            [int(tn), int(fp)],
                            [int(fn), int(tp)]
                        ],
                        "labels": ["Normal", "Abnormal"]
                    }
                },
                
                "error_analysis": {
                    "total_errors": int(fp + fn),
                    "error_rate": float((fp + fn) / len(y_test) * 100),
                    "false_alarm_rate": float(fp / (tn + fp) * 100) if (tn + fp) > 0 else 0.0,
                    "missed_anomaly_rate": float(fn / (fn + tp) * 100) if (fn + tp) > 0 else 0.0
                }
            },
            
            "model_artifacts": {
                "model_path": str(settings.MODEL_PATH),
                "scaler_path": str(settings.SCALER_PATH),
                "training_reference_path": str(settings.TRAINING_REFERENCE_PATH)
            },
            
            "production_readiness": {
                "overall_score": "EXCELLENT" if metrics['accuracy'] > 0.95 else "GOOD" if metrics['accuracy'] > 0.90 else "NEEDS_IMPROVEMENT",
                "recommendations": [
                    "Model is production-ready" if metrics['accuracy'] > 0.95 else "Consider retraining",
                    f"Zero false alarms achieved" if fp == 0 else f"{fp} false alarms detected",
                    f"Catches {metrics['recall']*100:.1f}% of anomalies",
                    f"Overall accuracy: {metrics['accuracy']*100:.2f}%"
                ]
            }
        }
        
        # Save to JSON
        report_path = settings.MODELS_DIR / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Training report saved to: {report_path}")
        logger.info(f"  - Dataset: {report['dataset_info']['total_samples']} samples")
        logger.info(f"  - Test Accuracy: {metrics['accuracy']*100:.2f}%")
        logger.info(f"  - Test F1-Score: {metrics['f1_score']*100:.2f}%")
        logger.info("="*80)

    def train(self, X: np.ndarray, y: np.ndarray, file_paths: List[str]) -> Dict:
        """Train model with proper stratified validation"""
        logger.info("="*80)
        logger.info("MODEL TRAINING")
        logger.info("="*80)
        
        # Two-step stratified split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=settings.TEST_SIZE,
            stratify=y, 
            random_state=settings.RANDOM_STATE,
            shuffle=True
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=settings.VAL_SIZE,
            stratify=y_train_val,
            random_state=settings.RANDOM_STATE,
            shuffle=True
        )
        
        # Log split info
        logger.info(f"Data split:")
        logger.info(f"  Training:   {len(X_train)} samples (N={np.sum(y_train==0)}, A={np.sum(y_train==1)})")
        logger.info(f"  Validation: {len(X_val)} samples (N={np.sum(y_val==0)}, A={np.sum(y_val==1)})")
        logger.info(f"  Test:       {len(X_test)} samples (N={np.sum(y_test==0)}, A={np.sum(y_test==1)})")
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save training reference
        X_all_scaled = self.scaler.transform(X)
        self.save_training_reference(X_all_scaled, y, file_paths)
        
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
            validation_data=(X_val_scaled, y_val),
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
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, y_pred_prob)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'y_pred_prob': y_pred_prob,  # For JSON report
            'y_pred': y_pred  # For JSON report
        }
        
        # Log metrics
        for metric, value in metrics.items():
            if metric not in ['y_pred_prob', 'y_pred']:
                logger.info(f"{metric.upper()}: {value:.4f} ({value*100:.2f}%)")
        
        # Save scaler
        with open(settings.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {settings.SCALER_PATH}")
        
        # ✅ NEW: Save comprehensive training report
        self.save_training_report(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            metrics, history
        )
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform k-fold cross-validation"""
        from sklearn.model_selection import StratifiedKFold
        
        logger.info("="*80)
        logger.info(f"CROSS-VALIDATION ({settings.CV_FOLDS}-Fold)")
        logger.info("="*80)
        
        kfold = StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True, 
                                random_state=settings.RANDOM_STATE)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            logger.info(f"Fold {fold}/{settings.CV_FOLDS}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale
            scaler_fold = StandardScaler()
            X_train_fold = scaler_fold.fit_transform(X_train_fold)
            X_val_fold = scaler_fold.transform(X_val_fold)
            
            # Train
            model_builder = PumpNetModel(X_train_fold.shape[1], settings.LEARNING_RATE)
            model = model_builder.build()
            
            model.fit(
                X_train_fold, y_train_fold,
                epochs=100,
                batch_size=settings.BATCH_SIZE,
                verbose=0
            )
            
            # Evaluate
            y_pred_prob = model.predict(X_val_fold, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_prob))
        
        # Summary
        logger.info("CV Results:")
        for metric, scores in cv_scores.items():
            mean = np.mean(scores)
            std = np.std(scores)
            logger.info(f"  {metric.upper()}: {mean:.4f} ± {std:.4f}")
        
        return cv_scores
    
    def run(self) -> Dict:
        """Run complete training pipeline"""
        normal_audio, abnormal_audio, sample_rate, file_paths = self.load_data()
        X, y, all_paths = self.prepare_data(normal_audio, abnormal_audio, sample_rate, file_paths)
        
        # Add cross-validation
        if settings.USE_CROSS_VALIDATION:
            cv_results = self.cross_validate(X, y)
        
        metrics = self.train(X, y, all_paths)
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