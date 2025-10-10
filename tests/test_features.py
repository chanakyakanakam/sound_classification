# Path: /home/chanakya/sound_classification/tests/test_features.py
import pytest
import numpy as np
from src.feature_extraction import AudioFeatureExtractor

def test_feature_extractor_initialization():
    extractor = AudioFeatureExtractor(n_mfcc=13)
    assert extractor.n_mfcc == 13

def test_extract_mfccs():
    extractor = AudioFeatureExtractor(n_mfcc=13)
    audio = np.random.randn(16000)  # 1 second at 16kHz
    sr = 16000
    
    mfccs = extractor.extract_mfccs(audio, sr)
    assert len(mfccs) == 13
    assert isinstance(mfccs, np.ndarray)

def test_extract_all_features():
    extractor = AudioFeatureExtractor(n_mfcc=13)
    audio = np.random.randn(16000)
    sr = 16000
    
    features = extractor.extract_all_features(audio, sr)
    assert len(features) == 18  # 13 MFCCs + 3 spectral + 2 temporal

    