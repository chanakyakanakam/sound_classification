"""
Path: /home/chanakya/sound_classification/src/explainability.py
AI Explainability Engine for Pump-Net
Provides similarity-based justification for predictions
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from config.config import settings
from utils.logger import logger


class ExplainabilityEngine:
    """
    Provides similarity-based explanations for model predictions
    Uses K-NN, clustering, and dimensionality reduction
    """
    
    def __init__(self, training_reference_path: Optional[Path] = None):
        """Initialize explainability engine"""
        self.training_reference_path = training_reference_path or settings.TRAINING_REFERENCE_PATH
        self.training_data: Optional[Dict] = None
        self.knn_model: Optional[NearestNeighbors] = None
        self.projection_model = None
        self.projected_features: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[Dict] = None
        
        # Load training reference
        self.load_training_reference()
        
        # Initialize models
        if self.training_data is not None:
            self._initialize_models()
    
    def load_training_reference(self):
        """Load training reference data"""
        if not self.training_reference_path.exists():
            logger.warning(f"Training reference not found: {self.training_reference_path}")
            logger.warning("Explainability features will not be available")
            self.training_data = None
            return
        
        try:
            logger.info("Loading training reference...")
            with open(self.training_reference_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Basic validation
            if loaded_data is None:
                logger.error("Training reference is None")
                self.training_data = None
                return
            
            if not isinstance(loaded_data, dict):
                logger.error(f"Training reference is not a dict, got {type(loaded_data)}")
                self.training_data = None
                return
            
            # Check required keys
            required_keys = ['features', 'labels']
            missing_keys = [k for k in required_keys if k not in loaded_data]
            if missing_keys:
                logger.error(f"Training reference missing keys: {missing_keys}")
                self.training_data = None
                return
            
            # Validate data types and shapes
            features = loaded_data['features']
            labels = loaded_data['labels']
            
            if not isinstance(features, np.ndarray):
                logger.error(f"Features is not numpy array, got {type(features)}")
                self.training_data = None
                return
            
            if not isinstance(labels, np.ndarray):
                logger.error(f"Labels is not numpy array, got {type(labels)}")
                self.training_data = None
                return
            
            if len(features.shape) != 2:
                logger.error(f"Features should be 2D array, got shape {features.shape}")
                self.training_data = None
                return
            
            if len(labels.shape) != 1:
                logger.error(f"Labels should be 1D array, got shape {labels.shape}")
                self.training_data = None
                return
            
            if features.shape[0] != labels.shape[0]:
                logger.error(f"Feature rows ({features.shape[0]}) != label count ({labels.shape[0]})")
                self.training_data = None
                return
            
            # All validation passed - store data as-is (no corruption!)
            self.training_data = loaded_data
            
            logger.info(f"✅ Loaded training reference: {len(features)} samples, {features.shape[1]} features")
            logger.info(f"   Normal: {int(np.sum(labels == 0))}, Abnormal: {int(np.sum(labels == 1))}")
            
        except Exception as e:
            logger.error(f"Failed to load training reference: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.training_data = None
    
    def _get_adaptive_k(self) -> int:
        """Determine optimal K based on dataset size"""
        if self.training_data is None:
            return settings.KNN_NEIGHBORS
        
        n_samples = len(self.training_data['features'])
        k = int(np.sqrt(n_samples))
        k = max(1, min(k, 10))
        
        if k % 2 == 0 and k > 1:
            k += 1
        
        return min(3, k)
    
    def _initialize_models(self):
        """Initialize KNN, projection, and clustering models"""
        if self.training_data is None:
            logger.error("Cannot initialize models: training_data is None")
            return
        
        try:
            X = self.training_data['features']
            y = self.training_data['labels']
            
            logger.info("Initializing explainability models...")
            
            # Determine K
            if hasattr(settings, 'KNN_ADAPTIVE') and settings.KNN_ADAPTIVE:
                k_to_use = self._get_adaptive_k()
            else:
                k_to_use = settings.KNN_NEIGHBORS
            
            # Initialize K-NN
            self.knn_model = NearestNeighbors(n_neighbors=min(k_to_use, len(X)), metric='euclidean')
            self.knn_model.fit(X)
            logger.info(f"✅ KNN initialized (k={k_to_use})")
            
            # Initialize projection
            if settings.PROJECTION_METHOD == 'pca':
                self.projection_model = PCA(n_components=2, random_state=settings.RANDOM_STATE)
                self.projected_features = self.projection_model.fit_transform(X)
                logger.info("✅ PCA projection initialized")
            else:  # tsne
                perplexity = min(30, len(X) - 1)
                self.projection_model = TSNE(n_components=2, random_state=settings.RANDOM_STATE, 
                                            perplexity=perplexity)
                self.projected_features = self.projection_model.fit_transform(X)
                logger.info("✅ t-SNE projection initialized")
            
            # Initialize clustering
            if settings.CLUSTERING_METHOD == 'dbscan':
                clustering = DBSCAN(eps=settings.DBSCAN_EPS, min_samples=settings.DBSCAN_MIN_SAMPLES)
                self.cluster_labels = clustering.fit_predict(X)
                
                unique_labels = set(self.cluster_labels.tolist())
                if -1 in unique_labels:
                    unique_labels.remove(-1)
                
                self.cluster_centers = {}
                for label in unique_labels:
                    mask = self.cluster_labels == label
                    self.cluster_centers[int(label)] = np.mean(X[mask], axis=0)
                
                logger.info(f"✅ DBSCAN clustering initialized ({len(unique_labels)} clusters)")
            
            logger.info("✅ All explainability models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def find_similar_examples(self, features: np.ndarray, k: Optional[int] = None) -> Dict:
        """Find K most similar examples from training data"""
        if self.training_data is None or self.knn_model is None:
            logger.error("Cannot find similar examples: models not initialized")
            return {'error': 'Training reference not loaded'}
        
        try:
            k = k if k is not None else settings.KNN_NEIGHBORS
            X = self.training_data['features']
            n_samples = len(X)
            
            max_neighbors = min(n_samples, k * 4)
            distances, indices = self.knn_model.kneighbors(features.reshape(1, -1), 
                                                           n_neighbors=max_neighbors)
            
            distances = distances[0]
            indices = indices[0]
            
            similar_normal = []
            similar_abnormal = []
            
            for dist, idx in zip(distances, indices):
                label = int(self.training_data['labels'][int(idx)])
                similarity = float(1 / (1 + dist))
                
                example = {
                    'index': int(idx),
                    'label': 'Normal' if label == 0 else 'Abnormal',
                    'distance': float(dist),
                    'similarity': similarity
                }
                
                if label == 0 and len(similar_normal) < k:
                    similar_normal.append(example)
                elif label == 1 and len(similar_abnormal) < k:
                    similar_abnormal.append(example)
                
                if len(similar_normal) >= k and len(similar_abnormal) >= k:
                    break
            
            closest_overall = None
            if similar_normal and similar_abnormal:
                closest_overall = 'Normal' if similar_normal[0]['distance'] < similar_abnormal[0]['distance'] else 'Abnormal'
            elif similar_normal:
                closest_overall = 'Normal'
            elif similar_abnormal:
                closest_overall = 'Abnormal'
            
            return {
                'similar_normal': similar_normal,
                'similar_abnormal': similar_abnormal,
                'closest_overall': closest_overall
            }
            
        except Exception as e:
            logger.error(f"Error finding similar examples: {e}")
            return {'error': str(e)}
    
    def calculate_cluster_distances(self, features: np.ndarray) -> Dict:
        """Calculate distances to cluster centers"""
        if self.training_data is None:
            logger.error("Cannot calculate distances: training_data is None")
            return {'error': 'Training reference not loaded'}
        
        try:
            X = self.training_data['features']
            y = self.training_data['labels']
            
            normal_center = np.mean(X[y == 0], axis=0)
            abnormal_center = np.mean(X[y == 1], axis=0)
            
            dist_to_normal = float(euclidean(features, normal_center))
            dist_to_abnormal = float(euclidean(features, abnormal_center))
            
            total_dist = dist_to_normal + dist_to_abnormal
            relative_to_abnormal = dist_to_abnormal / total_dist if total_dist > 0 else 0.5
            
            return {
                'distance_to_normal': dist_to_normal,
                'distance_to_abnormal': dist_to_abnormal,
                'relative_to_normal': float(1 - relative_to_abnormal),
                'relative_to_abnormal': float(relative_to_abnormal),
                'closer_to': 'Normal' if dist_to_normal < dist_to_abnormal else 'Abnormal',
                'distance_ratio': float(dist_to_abnormal / dist_to_normal) if dist_to_normal > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error calculating distances: {e}")
            return {'error': str(e)}
    
    def project_to_2d(self, features: np.ndarray) -> Tuple[float, float]:
        """Project features to 2D space"""
        if self.projection_model is None or self.projected_features is None:
            return (0.0, 0.0)
        
        try:
            if isinstance(self.projection_model, PCA):
                projected = self.projection_model.transform(features.reshape(1, -1))
                return (float(projected[0][0]), float(projected[0][1]))
            else:  # t-SNE
                if self.knn_model is not None:
                    distances, indices = self.knn_model.kneighbors(features.reshape(1, -1), n_neighbors=5)
                    weights = 1.0 / (distances[0] + 1e-6)
                    weights = weights / np.sum(weights)
                    projected = np.sum(self.projected_features[indices[0]] * weights.reshape(-1, 1), axis=0)
                    return (float(projected[0]), float(projected[1]))
            return (0.0, 0.0)
        except Exception as e:
            logger.error(f"Error projecting to 2D: {e}")
            return (0.0, 0.0)
    
    def get_visualization_data(self) -> Dict:
        """Get data for 2D visualization"""
        if self.training_data is None or self.projected_features is None:
            return {'error': 'Training reference not loaded'}
        
        return {
            'x': self.projected_features[:, 0].tolist(),
            'y': self.projected_features[:, 1].tolist(),
            'labels': self.training_data['labels'].tolist(),
            'projection_method': settings.PROJECTION_METHOD
        }
    
    def get_feature_importance(self, features: np.ndarray) -> List[Dict]:
        """Calculate feature importance based on deviation from normal"""
        if self.training_data is None:
            logger.error("Cannot get feature importance: training_data is None")
            return []
        
        try:
            X = self.training_data['features']
            y = self.training_data['labels']
            
            normal_mean = np.mean(X[y == 0], axis=0)
            normal_std = np.std(X[y == 0], axis=0) + 1e-6
            
            deviations = np.abs((features - normal_mean) / normal_std)
            
            feature_names = self._get_feature_names()
            
            importance = []
            for i, (name, dev) in enumerate(zip(feature_names, deviations)):
                importance.append({
                    'feature_name': name,
                    'deviation': float(dev),
                    'value': float(features[i]),
                    'normal_mean': float(normal_mean[i]),
                    'normal_std': float(normal_std[i])
                })
            
            importance.sort(key=lambda x: x['deviation'], reverse=True)
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return []
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names"""
        names = []
        
        if settings.EXTRACT_MFCCS:
            names.extend([f'MFCC_{i+1}' for i in range(settings.N_MFCC)])
        
        if settings.EXTRACT_SPECTRAL:
            names.extend(['Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Contrast'])
        
        if settings.EXTRACT_TEMPORAL:
            names.extend(['Zero_Crossing_Rate', 'Autocorrelation'])
        
        return names
    
    def explain_prediction(self, features: np.ndarray) -> Dict:
        """Generate comprehensive explanation for a prediction"""
        if self.training_data is None:
            logger.error("Cannot explain prediction: training_data is None")
            return {'error': 'Training reference not loaded', 'available': False}
        
        try:
            logger.info("Generating explainability data...")
            
            explanation = {
                'available': True,
                'similar_examples': self.find_similar_examples(features),
                'cluster_distances': self.calculate_cluster_distances(features),
                'feature_importance': self.get_feature_importance(features),
                'projection_2d': self.project_to_2d(features),
                'visualization_data': self.get_visualization_data()
            }
            
            logger.info("✅ Explainability data generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e), 'available': False}