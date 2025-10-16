"""
Path: /home/chanakya/sound_classification/src/data_preprocessing.py
Data Preprocessing Module
Handles audio loading and data augmentation
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm
from config.config import settings
from utils.logger import logger

class AudioDataLoader:
    """Load and preprocess audio data"""
    
    def __init__(self, sample_rate: Optional[int] = None):
        self.sample_rate = sample_rate
        logger.info(f"AudioDataLoader initialized with sample_rate={sample_rate}")
    
    def load_audio_file(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load single audio file
        
        Returns:
            Tuple of (audio, sample_rate) - sample_rate is guaranteed to be int
        """
        try:
            logger.debug(f"Loading audio file: {file_path}")
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Fix: Ensure sr is int
            sample_rate_int = int(sr)
            
            logger.debug(f"Loaded audio: duration={len(audio)/sample_rate_int:.2f}s, sr={sample_rate_int}Hz")
            return audio, sample_rate_int
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def load_directory(self, directory: Path, label: int, 
                      max_files: Optional[int] = None) -> Tuple[List[np.ndarray], List[int], int]:
        """
        Load all audio files from directory
        
        Returns:
            Tuple of (audio_list, labels, sample_rate) - sample_rate is guaranteed to be int
        """
        audio_files = []
        labels = []
        sample_rate: Optional[int] = None
        
        wav_files = list(directory.glob("*.wav"))
        if max_files:
            wav_files = wav_files[:max_files]
        
        logger.info(f"Loading {len(wav_files)} files from {directory}")
        
        for file_path in tqdm(wav_files, desc=f"Loading label={label}"):
            try:
                audio, sr = self.load_audio_file(file_path)
                
                if sample_rate is None:
                    sample_rate = sr
                
                audio_files.append(audio)
                labels.append(label)
                
            except Exception as e:
                logger.warning(f"Skipping file {file_path}: {e}")
        
        # Fix: Ensure sample_rate is not None
        if sample_rate is None:
            raise ValueError(f"No valid audio files found in {directory}")
        
        logger.info(f"Successfully loaded {len(audio_files)} files")
        return audio_files, labels, sample_rate

class AudioAugmentor:
    """Apply data augmentation to audio signals"""
    
    def __init__(self, noise_factor: float = 0.005, shift_max: float = 0.2):
        self.noise_factor = noise_factor
        self.shift_max = shift_max
        logger.info(f"AudioAugmentor initialized: noise={noise_factor}, shift={shift_max}")
    
    def augment(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply augmentation to audio"""
        augmented = audio.copy()
        
        # Time shifting
        shift_samples = int(np.random.uniform(-self.shift_max, self.shift_max) * sample_rate)
        augmented = np.roll(augmented, shift_samples)
        
        # Add Gaussian noise
        noise = np.random.randn(len(augmented))
        augmented = augmented + self.noise_factor * noise
        
        logger.debug(f"Applied augmentation: shift={shift_samples} samples")
        return augmented
    
    