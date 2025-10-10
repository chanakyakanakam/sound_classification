"""
Path: /home/chanakya/sound_classification/src/model.py
Model Architecture Module
Defines Pump-Net neural network architecture
"""
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras import Sequential                 
from tensorflow.keras.layers import Dense, Dropout     
from config.config import settings
from utils.logger import logger

class PumpNetModel:
    """Pump-Net neural network for anomaly detection"""
    
    def __init__(self, input_dim: int, learning_rate: float = 0.001):
        """
        Initialize model
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = None
        logger.info(f"PumpNetModel initialized: input_dim={input_dim}, lr={learning_rate}")
    
    def build(self) -> Sequential:
        """
        Build model architecture
        """
        logger.info("Building Pump-Net architecture")
        
        model = Sequential(name='Pump_Net')
        model.add(Input(shape=(self.input_dim,)))
        
        # Hidden Layer 1
        model.add(Dense(settings.HIDDEN_UNITS_1, activation='relu', name='dense_1'))
        model.add(Dropout(settings.DROPOUT_RATE, name='dropout_1'))
        
        # Hidden Layer 2
        model.add(Dense(settings.HIDDEN_UNITS_2, activation='relu', name='dense_2'))
        model.add(Dropout(settings.DROPOUT_RATE, name='dropout_2'))
        
        # Output Layer
        model.add(Dense(1, activation='sigmoid', name='output'))
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),     
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),                           
                tf.keras.metrics.Recall(name='recall'),                                
                tf.keras.metrics.AUC(name='auc')                                          
            ]
        )
        
        self.model = model
        logger.info(f"Model built successfully: {model.count_params():,} parameters")
        
        return model
    
    def get_model(self) -> Sequential:
        """Get built model"""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model