"""
# Path: /home/chanakya/sound_classification/tests/test_model.py
Unit Tests for Pump-Net Model
Tests model architecture, training, and inference
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import tensorflow as tf
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PumpNetModel
from src.inference import PumpNetInference
from config.config import settings


class TestPumpNetModel:
    """Test cases for PumpNetModel class"""
    
    def test_model_initialization(self):
        """Test model initialization with correct parameters"""
        input_dim = 18
        learning_rate = 0.001
        
        model_builder = PumpNetModel(input_dim=input_dim, learning_rate=learning_rate)
        
        assert model_builder.input_dim == input_dim
        assert model_builder.learning_rate == learning_rate
        assert model_builder.model is None  # Not built yet
    
    def test_model_build(self):
        """Test model architecture building"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Check model is Sequential
        assert isinstance(model, tf.keras.Sequential)
        
        # Check model has correct input shape
        assert model.input_shape == (None, 18)
        
        # Check model has correct output shape
        assert model.output_shape == (None, 1)
        
        # Check model is compiled
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_model_layers(self):
        """Test model has correct layer structure"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Check number of layers (Dense + Dropout + Dense + Dropout + Dense = 5)
        assert len(model.layers) == 5
        
        # Check layer types
        assert isinstance(model.layers[0], tf.keras.layers.Dense)  # dense_1
        assert isinstance(model.layers[1], tf.keras.layers.Dropout)  # dropout_1
        assert isinstance(model.layers[2], tf.keras.layers.Dense)  # dense_2
        assert isinstance(model.layers[3], tf.keras.layers.Dropout)  # dropout_2
        assert isinstance(model.layers[4], tf.keras.layers.Dense)  # output
    
    def test_model_layer_units(self):
        """Test layer units match configuration"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Check first hidden layer units
        assert model.layers[0].units == settings.HIDDEN_UNITS_1
        
        # Check second hidden layer units
        assert model.layers[2].units == settings.HIDDEN_UNITS_2
        
        # Check output layer units
        assert model.layers[4].units == 1
    
    def test_model_activation_functions(self):
        """Test activation functions are correct"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Check hidden layers use ReLU
        assert model.layers[0].activation.__name__ == 'relu'
        assert model.layers[2].activation.__name__ == 'relu'
        
        # Check output layer uses sigmoid
        assert model.layers[4].activation.__name__ == 'sigmoid'
    
    def test_model_dropout_rate(self):
        """Test dropout layers have correct rate"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Check dropout rates
        assert model.layers[1].rate == settings.DROPOUT_RATE
        assert model.layers[3].rate == settings.DROPOUT_RATE
    
    def test_model_optimizer(self):
        """Test model uses Adam optimizer with correct learning rate"""
        learning_rate = 0.001
        model_builder = PumpNetModel(input_dim=18, learning_rate=learning_rate)
        model = model_builder.build()
        
        # Check optimizer type
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
        
        # Check learning rate
        assert float(model.optimizer.learning_rate.numpy()) == learning_rate
    
    def test_model_loss_function(self):
        """Test model uses binary crossentropy loss"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Check loss function
        assert model.loss == 'binary_crossentropy'
    
    def test_model_metrics(self):
        """Test model tracks correct metrics"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Get metric names
        metric_names = [m.name for m in model.metrics]
        
        # Check required metrics are present
        assert 'accuracy' in metric_names
        assert 'precision' in metric_names
        assert 'recall' in metric_names
        assert 'auc' in metric_names
    
    def test_model_parameter_count(self):
        """Test model has reasonable number of parameters"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        total_params = model.count_params()
        
        # Model should have parameters
        assert total_params > 0
        
        # Calculate expected parameters
        # Layer 1: (18 * 64) + 64 = 1216
        # Layer 2: (64 * 64) + 64 = 4160
        # Layer 3: (64 * 1) + 1 = 65
        # Total: 5441
        expected_params = (18 * 64 + 64) + (64 * 64 + 64) + (64 * 1 + 1)
        assert total_params == expected_params
    
    def test_model_prediction_shape(self):
        """Test model output shape is correct"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Create dummy input
        X_test = np.random.randn(10, 18)
        
        # Predict
        predictions = model.predict(X_test, verbose=0)
        
        # Check output shape
        assert predictions.shape == (10, 1)
    
    def test_model_prediction_range(self):
        """Test model predictions are in valid probability range [0, 1]"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Create dummy input
        X_test = np.random.randn(100, 18)
        
        # Predict
        predictions = model.predict(X_test, verbose=0)
        
        # Check all predictions are in [0, 1]
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
    
    def test_get_model_before_build_raises_error(self):
        """Test getting model before building raises error"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        
        with pytest.raises(ValueError, match="Model not built"):
            model_builder.get_model()
    
    def test_get_model_after_build(self):
        """Test getting model after building succeeds"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model_builder.build()
        
        # Should not raise error
        model = model_builder.get_model()
        assert model is not None
    
    def test_model_training_single_epoch(self):
        """Test model can train for one epoch"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Create dummy training data
        X_train = np.random.randn(50, 18)
        y_train = np.random.randint(0, 2, size=(50,))
        
        # Train for 1 epoch
        history = model.fit(X_train, y_train, epochs=1, verbose=0)
        
        # Check history contains loss
        assert 'loss' in history.history
        assert len(history.history['loss']) == 1
    
    def test_model_save_and_load(self):
        """Test model can be saved and loaded"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            model.save(tmp_path)
            
            # Load model
            loaded_model = tf.keras.models.load_model(tmp_path)
            
            # Check loaded model has same architecture
            assert loaded_model.input_shape == model.input_shape
            assert loaded_model.output_shape == model.output_shape
            assert len(loaded_model.layers) == len(model.layers)
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


class TestPumpNetInference:
    """Test cases for PumpNetInference class"""
    
    @pytest.fixture
    def mock_model_and_scaler(self):
        """Create mock model and scaler for testing"""
        # Create a simple model
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Create temporary files
        model_tmp = tempfile.NamedTemporaryFile(suffix='.keras', delete=False)
        scaler_tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        
        model.save(model_tmp.name)
        
        # Create mock scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 18))
        
        import pickle
        with open(scaler_tmp.name, 'wb') as f:
            pickle.dump(scaler, f)
        
        yield Path(model_tmp.name), Path(scaler_tmp.name)
        
        # Cleanup
        Path(model_tmp.name).unlink(missing_ok=True)
        Path(scaler_tmp.name).unlink(missing_ok=True)
    
    def test_inference_initialization(self, mock_model_and_scaler):
        """Test inference pipeline initialization"""
        model_path, scaler_path = mock_model_and_scaler
        
        inference = PumpNetInference(model_path=model_path, scaler_path=scaler_path)
        
        assert inference.model is not None
        assert inference.scaler is not None
        assert inference.feature_extractor is not None
        assert inference.data_loader is not None
    
    def test_predict_from_audio(self, mock_model_and_scaler):
        """Test prediction from audio array"""
        model_path, scaler_path = mock_model_and_scaler
        inference = PumpNetInference(model_path=model_path, scaler_path=scaler_path)
        
        # Create dummy audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        sample_rate = 16000
        
        # Predict
        result = inference.predict_from_audio(audio, sample_rate)
        
        # Check result structure
        assert 'prediction' in result
        assert 'prediction_class' in result
        assert 'probability_abnormal' in result
        assert 'probability_normal' in result
        assert 'confidence' in result
        assert 'is_confident' in result
        assert 'threshold_used' in result
        
        # Check value types
        assert isinstance(result['prediction'], str)
        assert result['prediction'] in ['Normal', 'Abnormal']
        assert isinstance(result['prediction_class'], int)
        assert result['prediction_class'] in [0, 1]
        assert 0 <= result['probability_abnormal'] <= 1
        assert 0 <= result['probability_normal'] <= 1
        assert 0 <= result['confidence'] <= 1
        assert isinstance(result['is_confident'], bool)
    
    def test_prediction_probabilities_sum_to_one(self, mock_model_and_scaler):
        """Test prediction probabilities sum to 1"""
        model_path, scaler_path = mock_model_and_scaler
        inference = PumpNetInference(model_path=model_path, scaler_path=scaler_path)
        
        audio = np.random.randn(16000)
        sample_rate = 16000
        
        result = inference.predict_from_audio(audio, sample_rate)
        
        # Check probabilities sum to ~1.0
        prob_sum = result['probability_normal'] + result['probability_abnormal']
        assert abs(prob_sum - 1.0) < 1e-6
    
    def test_confidence_calculation(self, mock_model_and_scaler):
        """Test confidence is calculated correctly"""
        model_path, scaler_path = mock_model_and_scaler
        inference = PumpNetInference(model_path=model_path, scaler_path=scaler_path)
        
        audio = np.random.randn(16000)
        sample_rate = 16000
        
        result = inference.predict_from_audio(audio, sample_rate)
        
        # Confidence should be max of two probabilities
        expected_confidence = max(result['probability_normal'], result['probability_abnormal'])
        assert abs(result['confidence'] - expected_confidence) < 1e-6
    
    def test_prediction_consistency(self, mock_model_and_scaler):
        """Test predictions are consistent for same input"""
        model_path, scaler_path = mock_model_and_scaler
        inference = PumpNetInference(model_path=model_path, scaler_path=scaler_path)
        
        audio = np.random.randn(16000)
        sample_rate = 16000
        
        # Predict twice
        result1 = inference.predict_from_audio(audio, sample_rate)
        result2 = inference.predict_from_audio(audio, sample_rate)
        
        # Results should be identical
        assert result1['prediction'] == result2['prediction']
        assert result1['prediction_class'] == result2['prediction_class']
        assert abs(result1['probability_abnormal'] - result2['probability_abnormal']) < 1e-6


class TestModelIntegration:
    """Integration tests for complete model pipeline"""
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline from features to output"""
        # Build model
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Create dummy data
        X_train = np.random.randn(100, 18)
        y_train = np.random.randint(0, 2, size=(100,))
        
        # Train for a few epochs
        model.fit(X_train, y_train, epochs=3, verbose=0)
        
        # Create test data
        X_test = np.random.randn(10, 18)
        
        # Predict
        predictions = model.predict(X_test, verbose=0)
        
        # Verify predictions
        assert predictions.shape == (10, 1)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
    
    def test_model_improves_with_training(self):
        """Test model loss decreases with training"""
        model_builder = PumpNetModel(input_dim=18, learning_rate=0.001)
        model = model_builder.build()
        
        # Create separable training data
        X_train = np.vstack([
            np.random.randn(50, 18) - 2,  # Class 0
            np.random.randn(50, 18) + 2   # Class 1
        ])
        y_train = np.array([0] * 50 + [1] * 50)
        
        # Train and track loss
        history = model.fit(X_train, y_train, epochs=20, verbose=0)
        
        # Initial loss should be higher than final loss
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])