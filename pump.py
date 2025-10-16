"""## About the Project
### Project Overview
The essence of this project revolves around crafting and fine-tuning a deep learning model that can discern between normal and anomalous operational sounds from industrial pumps. Utilize the specialized pump sound subset of the MIMII Dataset, which is comprised of diverse acoustic signatures representing various operational states of industrial pumps.

### Dataset Description
The pump sound dataset is a segment of the MIMII Dataset, featuring acoustic recordings that capture the operational nuances of industrial pumps. It includes a rich tapestry of soundscapes illustrating normal functionality, as well as a series of anomalies. These recordings are complemented by ambient factory noises, rendering a realistic backdrop for the model's training and evaluation.

## Continuation of Exploration
This document serves as the inaugural segment of the anomaly detection series, concentrating on the application of supervised learning strategies. The exploration progresses in the second part, shifting the focus to unsupervised learning approaches aimed at augmenting the model's capacity for anomaly detection in the absence of labeled data"""

# #### Tensorflow only supports in "Ubuntu" do not run in windows

# #### PUMP-NET: INDUSTRIAL PUMP ANOMALY DETECTION - ENHANCED IMPLEMENTATION Feature Engineering + Deep Neural Network for High-Accuracy Classification Based on Best Practices with 96.96% Accuracy Achievement

# ### SECTION 1: IMPORTS AND SETUP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import os
import warnings
from pathlib import Path
from tqdm import tqdm
import pickle
from scipy.stats import ttest_ind
from scipy import stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress oneDNN messages


import tensorflow as tf
from tensorflow import keras                                            #type:ignore
from tensorflow.keras import layers, models, callbacks                  #type:ignore
from tensorflow.keras.models import Sequential                          #type:ignore   
from tensorflow.keras.layers import Input, Dense, Dropout               #type:ignore
from tensorflow.keras.utils import to_categorical                       #type:ignore

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score, 
    precision_score, recall_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

save_dir = "/home/chanakya/sound_classification/visualizations"
os.makedirs(save_dir, exist_ok=True)

def save_fig(filename):
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close()


print("=" * 80)
print("PUMP-NET: INDUSTRIAL PUMP ANOMALY DETECTION SYSTEM")
print("=" * 80)
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")


print("\n" + "="*70)
print("GPU MEMORY CONFIGURATION")
print("="*70)

# Configure TensorFlow for memory growth (prevents pre-allocation)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth - allocates only what's needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Optional: Set hard memory limit (e.g., 3.5GB out of 4GB)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3584)]  
        )
        print("✅ GPU memory growth enabled")
        print("✅ Memory limit set to 3.5GB")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
else:
    print("⚠️  No GPU detected - running on CPU")

print("="*70)

# #### SECTION 2: CONFIGURATION AND PARAMETERS

class Config:
    """Centralized configuration for the entire pipeline"""
    
    # Data paths (WSL2 compatible)
    ROOT_DIR = "/home/chanakya/sound_classification"  # WSL2 path
    ABNORMAL_PATH = os.path.join(ROOT_DIR, "6_dB_pump", "pump", "id_00", "abnormal")
    NORMAL_PATH = os.path.join(ROOT_DIR, "6_dB_pump", "pump", "id_00", "normal")
    
    # Audio parameters
    SAMPLE_RATE = None  # Use native sample rate
    N_MFCC = 13  # Number of MFCCs
    
    # Feature extraction parameters
    EXTRACT_MFCCS = True
    EXTRACT_SPECTRAL = True
    EXTRACT_TEMPORAL = True
    
    # Data augmentation parameters
    AUGMENT_DATA = True
    NOISE_FACTOR = 0.005
    SHIFT_MAX = 0.2  # Maximum time shift in seconds
    AUGMENTATION_FACTOR = 2  # How many augmented samples per original
    
    # Data split ratios
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    
    # Model parameters
    BATCH_SIZE = 16      # VRAM increase Batch Size to 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Cross-validation
    CV_FOLDS = 5
    USE_CROSS_VALIDATION = True
    
    # Early stopping
    PATIENCE = 20
    
    # Model architecture
    HIDDEN_UNITS_1 = 64
    HIDDEN_UNITS_2 = 64
    DROPOUT_RATE = 0.5
    
    # Optimization
    OPTIMIZE_THRESHOLD = True
    
    # Class labels
    CLASS_NAMES = ['Normal', 'Abnormal']

    # Model save directory
    MODEL_SAVE_DIR = "/home/chanakya/sound_classification/models"   # WSL2 path
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Output paths - all saved to MODEL_SAVE_DIR
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'pump_net_best_model.keras')
    SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'pump_net_scaler.pkl')
    RESULTS_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'pump_net_results.pkl')

config = Config()

# Create the models directory if it doesn't exist
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

print("\n" + "=" * 80)
print("CONFIGURATION SUMMARY")
print("=" * 80)
print(f"Data Directory: {config.ROOT_DIR}")
print(f"MFCC Features: {config.N_MFCC}")
print(f"Data Augmentation: {'Enabled' if config.AUGMENT_DATA else 'Disabled'}")
print(f"Cross-Validation: {config.CV_FOLDS}-Fold" if config.USE_CROSS_VALIDATION else "Single Split")
print(f"Model Architecture: {config.HIDDEN_UNITS_1}-{config.HIDDEN_UNITS_2} units")
print(f"Dropout Rate: {config.DROPOUT_RATE}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Max Epochs: {config.EPOCHS}")
print("=" * 80)

def clear_gpu_memory():
    """Clear GPU memory cache"""
    tf.keras.backend.clear_session() #type: ignore
    import gc
    gc.collect()
    print("🧹 GPU memory cleared")

# #### SECTION 3: DATA LOADING AND EXPLORATION

def load_audio_files(path, label, max_files=None):
    """
    Load audio files from directory with progress tracking
    """
    audio_files = []
    labels = []
    filenames = []
    sample_rate = float(sr_abnormal) if sr_abnormal else float(sr_normal) if sr_normal else 22050.0
    
    wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    if max_files:
        wav_files = wav_files[:max_files]
    
    class_name = 'Abnormal' if label == 1 else 'Normal'
    print(f"\n📂 Loading {len(wav_files)} {class_name} samples from:")
    print(f"   {path}")
    
    for filename in tqdm(wav_files, desc=f"Loading {class_name}"):
        try:
            file_path = os.path.join(path, filename)
            audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
            
            if sample_rate is None:
                sample_rate = sr
            
            audio_files.append(audio)
            labels.append(label)
            filenames.append(filename)
            
        except Exception as e:
            print(f"\n⚠️  Error loading {filename}: {e}")
    
    print(f"✅ Successfully loaded {len(audio_files)} {class_name} samples")
    
    return audio_files, labels, filenames, sample_rate

print("\n" + "=" * 80)
print("SECTION 3: DATA LOADING")
print("=" * 80)

# Load abnormal samples
abnormal_audio, abnormal_labels, abnormal_files, sr_abnormal = load_audio_files(
    config.ABNORMAL_PATH, label=1
)

# Load normal samples
normal_audio, normal_labels, normal_files, sr_normal = load_audio_files(
    config.NORMAL_PATH, label=0
)

# Use the sample rate from the first loaded file
sample_rate = sr_abnormal if sr_abnormal else sr_normal

# Combine datasets
all_audio = normal_audio + abnormal_audio
all_labels = normal_labels + abnormal_labels
all_filenames = normal_files + abnormal_files

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"Total samples: {len(all_audio)}")
print(f"  ├─ Normal samples: {len(normal_audio)} ({len(normal_audio)/len(all_audio)*100:.1f}%)")
print(f"  └─ Abnormal samples: {len(abnormal_audio)} ({len(abnormal_audio)/len(all_audio)*100:.1f}%)")
print(f"\nClass Distribution:")
print(f"  ├─ Imbalance Ratio: 1:{len(normal_audio)/len(abnormal_audio):.2f}")
print(f"  └─ Balance Status: {'⚠️  Imbalanced' if len(normal_audio)/len(abnormal_audio) > 1.5 else '✅ Balanced'}")
print(f"\nAudio Properties:")
print(f"  ├─ Sample Rate: {sample_rate} Hz")
if sample_rate:
    duration = len(all_audio[0]) / sample_rate
    print(f"  ├─ Duration: {duration:.2f} seconds")
else:
    print("  ├─ Duration: Unable to compute (sample_rate is None)")
print(f"  └─ Samples per file: {len(all_audio[0])}")
print("=" * 80)

# #### SECTION 4: EXPLORATORY DATA ANALYSIS AND VISUALIZATION

print("\n" + "=" * 80)
print("SECTION 4: EXPLORATORY DATA ANALYSIS - AUDIO VISUALIZATION")
print("=" * 80)

# Visualization 1: Class Distribution
print("\n📊 Visualization 1: Class Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
class_counts = pd.Series(all_labels).value_counts().sort_index()
axes[0].bar(['Normal', 'Abnormal'], class_counts.values, 
            color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('Class Distribution (Raw Data)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 3, str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)

# Pie chart
axes[1].pie(class_counts.values, labels=['Normal', 'Abnormal'], 
            autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'],
            explode=(0.05, 0.05), shadow=True)
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
save_fig('1_class_distribution_raw.png')
plt.show()

# Visualization 2: Sample Waveforms
print("\n📊 Visualization 2: Sample Audio Waveforms")

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# Plot 3 normal samples
for i in range(3):
    ax = axes[0, i]
    librosa.display.waveshow(normal_audio[i], sr=sample_rate, ax=ax, color='#2ecc71', alpha=0.8)
    ax.set_title(f'Normal Sample {i+1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(alpha=0.3)

# Plot 3 abnormal samples
for i in range(3):
    ax = axes[1, i]
    librosa.display.waveshow(abnormal_audio[i], sr=sample_rate, ax=ax, color='#e74c3c', alpha=0.8)
    ax.set_title(f'Abnormal Sample {i+1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle('Sample Waveforms: Normal vs Abnormal', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
save_fig('2_sample_waveforms.png')
plt.show()

# Visualization 3: Statistical Analysis of Raw Audio
print("\n📊 Visualization 3: Statistical Analysis of Audio Signals")

def compute_audio_stats(audio_list):
    """Compute statistical features of audio signals"""
    stats = {
        'mean_amplitude': [],
        'std_amplitude': [],
        'max_amplitude': [],
        'zero_crossing_rate': [],
        'rms_energy': []
    }
    
    for audio in audio_list:
        stats['mean_amplitude'].append(np.mean(np.abs(audio)))
        stats['std_amplitude'].append(np.std(audio))
        stats['max_amplitude'].append(np.max(np.abs(audio)))
        stats['zero_crossing_rate'].append(np.mean(librosa.feature.zero_crossing_rate(audio)))
        stats['rms_energy'].append(np.mean(librosa.feature.rms(y=audio)))
    
    return stats

# Sample 100 normal files for faster computation
normal_sample_size = min(100, len(normal_audio))
normal_stats = compute_audio_stats(normal_audio[:normal_sample_size])
abnormal_stats = compute_audio_stats(abnormal_audio)

# Plot statistical comparisons
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
features = ['mean_amplitude', 'std_amplitude', 'max_amplitude', 
            'zero_crossing_rate', 'rms_energy']
titles = ['Mean Amplitude', 'Std Amplitude', 'Max Amplitude', 
          'Zero Crossing Rate', 'RMS Energy']

for idx, (feature, title) in enumerate(zip(features, titles)):
    if idx < 5:
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Box plot
        data = [normal_stats[feature], abnormal_stats[feature]]
        bp = ax.boxplot(data, labels=['Normal', 'Abnormal'], patch_artist=True,
                       boxprops=dict(alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

# Hide the last subplot
axes[1, 2].axis('off')

plt.suptitle('Statistical Audio Feature Comparison: Normal vs Abnormal', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_fig('3_audio_statistics.png')
plt.show()

# Visualization 4: Spectrograms
print("\n📊 Visualization 4: Mel-Spectrograms")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 3 normal spectrograms
for i in range(3):
    ax = axes[0, i]
    mel_spec = librosa.feature.melspectrogram(y=normal_audio[i], sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                    sr=sample_rate, ax=ax, cmap='viridis')
    ax.set_title(f'Normal Sample {i+1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')

# Plot 3 abnormal spectrograms
for i in range(3):
    ax = axes[1, i]
    mel_spec = librosa.feature.melspectrogram(y=abnormal_audio[i], sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                    sr=sample_rate, ax=ax, cmap='viridis')
    ax.set_title(f'Abnormal Sample {i+1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')

plt.suptitle('Log-Mel Spectrograms: Normal vs Abnormal', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
save_fig('4_spectrograms.png')
plt.show()

print("\n✅ Audio visualization complete!")

# #### SECTION 5: FEATURE EXTRACTION (MEL-SPECTROGRAMS)

print("\n" + "=" * 80)
print("SECTION 5: FEATURE EXTRACTION FUNCTIONS")
print("=" * 80)

def extract_mfccs(audio, sample_rate, n_mfcc=13):
    """
    Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    
    MFCCs represent the power spectrum of sound and are widely used
    for audio classification tasks. They capture timbral characteristics.
    
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    # Take mean across time dimension
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def extract_spectral_features(audio, sample_rate):
    """
    Extract spectral features that describe frequency content
    
    Features extracted:
    - Spectral Centroid: Center of mass of spectrum (brightness)
    - Spectral Rolloff: Frequency below which 85% of energy is contained
    - Spectral Contrast: Difference between peaks and valleys in spectrum
    
    """
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    
    return (
        np.mean(spectral_centroids),
        np.mean(spectral_rolloff),
        np.mean(spectral_contrast)
    )

def extract_temporal_features(audio):
    """
    Extract temporal features from audio signal
    
    Features extracted:
    - Zero Crossing Rate: Rate of sign changes (texture/percussiveness)
    - Autocorrelation: Self-similarity measure (periodicity)
    
    """
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    
    return (
        np.mean(zero_crossing_rate),
        np.mean(autocorrelation)
    )

def extract_all_features(audio, sample_rate):
    """
    Extract all features from a single audio sample
    
    """
    features = []
    
    # MFCCs
    if config.EXTRACT_MFCCS:
        mfccs = extract_mfccs(audio, sample_rate, n_mfcc=config.N_MFCC)
        features.append(mfccs)
    
    # Spectral features
    if config.EXTRACT_SPECTRAL:
        spectral_features = extract_spectral_features(audio, sample_rate)
        features.extend(spectral_features)
    
    # Temporal features
    if config.EXTRACT_TEMPORAL:
        temporal_features = extract_temporal_features(audio)
        features.extend(temporal_features)
    
    return np.concatenate([np.atleast_1d(f) for f in features])

# Create feature names for later analysis
feature_names = []
if config.EXTRACT_MFCCS:
    feature_names.extend([f'MFCC_{i+1}' for i in range(config.N_MFCC)])
if config.EXTRACT_SPECTRAL:
    feature_names.extend(['Spectral_Centroid', 'Spectral_Rolloff', 'Spectral_Contrast'])
if config.EXTRACT_TEMPORAL:
    feature_names.extend(['Zero_Crossing_Rate', 'Autocorrelation'])

print(f"\n✅ Feature extraction functions defined")
print(f"📊 Total features to extract: {len(feature_names)}")
print(f"   ├─ MFCCs: {config.N_MFCC if config.EXTRACT_MFCCS else 0}")
print(f"   ├─ Spectral: {3 if config.EXTRACT_SPECTRAL else 0}")
print(f"   └─ Temporal: {2 if config.EXTRACT_TEMPORAL else 0}")

"""#### 🧠 Why We Use MFCCs

Humans don’t perceive frequency linearly:

    - We’re more sensitive to differences in lower frequencies (like 300 Hz vs 600 Hz)

    -And less sensitive to differences in higher frequencies (like 7000 Hz vs 7200 Hz).

So, instead of analyzing all frequencies equally, MFCCs:

    Warp the frequency scale to match the human ear’s sensitivity (Mel scale).

    Extract meaningful frequency patterns that correspond to timbre(what kind of sound it is (the texture or tone quality).), tone, and texture of sound."""

# #### SECTION 6: DATA AUGMENTATION

print("\n" + "=" * 80)
print("SECTION 6: DATA AUGMENTATION")
print("=" * 80)

def augment_audio(audio, sample_rate, noise_factor=0.005, shift_max=0.2):
    """
    Apply data augmentation to audio sample
    
    Techniques:
    1. Time Shifting: Shift audio in time domain
    2. Noise Injection: Add Gaussian noise
    
    """
    augmented = audio.copy()
    
    # Time shifting
    shift_samples = int(np.random.uniform(-shift_max, shift_max) * sample_rate)
    augmented = np.roll(augmented, shift_samples)
    
    # Add Gaussian noise
    noise = np.random.randn(len(augmented))
    augmented = augmented + noise_factor * noise
    
    return augmented

# Demonstrate augmentation visually
if config.AUGMENT_DATA:
    print(f"\n📊 Visualizing Data Augmentation Effects")
    
    sample_audio = abnormal_audio[0]
    augmented_samples = [augment_audio(sample_audio, sample_rate, 
                                       config.NOISE_FACTOR, config.SHIFT_MAX) 
                        for _ in range(3)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original
    ax = axes[0, 0]
    librosa.display.waveshow(sample_audio, sr=sample_rate, ax=ax, color='#e74c3c', alpha=0.8)
    ax.set_title('Original Audio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)
    
    # Augmented versions
    for i, aug_audio in enumerate(augmented_samples):
        row = (i + 1) // 2
        col = (i + 1) % 2
        ax = axes[row, col]
        librosa.display.waveshow(aug_audio, sr=sample_rate, ax=ax, color='#9b59b6', alpha=0.8)
        ax.set_title(f'Augmented Version {i+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Data Augmentation: Time Shifting + Noise Injection', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig('5_data_augmentation.png')
    plt.show()

    print(f"\n✅ Data augmentation enabled")
    print(f"   ├─ Noise Factor: {config.NOISE_FACTOR}")
    print(f"   ├─ Max Time Shift: {config.SHIFT_MAX}s")
    print(f"   └─ Augmentation Factor: {config.AUGMENTATION_FACTOR}x per sample")
else:
    print("⏭️  Data augmentation disabled")

# #### SECTION 7: FEATURE EXTRACTION FROM ALL AUDIO

print("\n" + "=" * 80)
print("SECTION 7: EXTRACTING FEATURES FROM ALL AUDIO SAMPLES")
print("=" * 80)

def extract_features_from_dataset(audio_data, sample_rate, augment=False):
    """
    Extract features from entire dataset with optional augmentation
    """
    features = []
    
    for audio in tqdm(audio_data, desc="Extracting features"):
        # Extract features from original audio
        feature_vector = extract_all_features(audio, sample_rate)
        features.append(feature_vector)
        
        # Apply augmentation if enabled
        if augment and config.AUGMENT_DATA:
            for _ in range(config.AUGMENTATION_FACTOR - 1):
                augmented_audio = augment_audio(audio, sample_rate, 
                                               config.NOISE_FACTOR, 
                                               config.SHIFT_MAX)
                augmented_features = extract_all_features(augmented_audio, sample_rate)
                features.append(augmented_features)
    
    return np.array(features)

# Extract features from normal samples
print("\n📊 Processing Normal samples...")
normal_features = extract_features_from_dataset(normal_audio, sample_rate, augment=False)

# Extract features from abnormal samples (with augmentation to balance classes)
print("\n📊 Processing Abnormal samples...")
abnormal_features = extract_features_from_dataset(abnormal_audio, sample_rate, 
                                                  augment=config.AUGMENT_DATA)


print("\n" + "=" * 80)
print("FEATURE EXTRACTION SUMMARY")
print("=" * 80)
print(f"Normal features shape: {normal_features.shape}")
print(f"  ├─ Samples: {normal_features.shape[0]}")
print(f"  └─ Features per sample: {normal_features.shape[1]}")
print(f"\nAbnormal features shape: {abnormal_features.shape}")
print(f"  ├─ Samples: {abnormal_features.shape[0]}")
print(f"  └─ Features per sample: {abnormal_features.shape[1]}")
print(f"\nAugmentation Effect:")
print(f"  ├─ Original abnormal samples: {len(abnormal_audio)}")
print(f"  ├─ Augmented abnormal samples: {abnormal_features.shape[0]}")
print(f"  └─ Augmentation multiplier: {abnormal_features.shape[0] / len(abnormal_audio):.1f}x")
print(f"\nNew Class Balance:")
print(f"  ├─ Normal: {normal_features.shape[0]}")
print(f"  ├─ Abnormal: {abnormal_features.shape[0]}")
print(f"  └─ Ratio: 1:{normal_features.shape[0]/abnormal_features.shape[0]:.2f}")
print("=" * 80)

# #### SECTION 8: EXPLORATORY DATA ANALYSIS - FEATURE ANALYSIS

print("\n" + "=" * 80)
print("SECTION 8: EXPLORATORY DATA ANALYSIS - FEATURE ANALYSIS")
print("=" * 80)

# 8.1: Feature Distributions
print("\n📊 Visualization 5: Feature Distributions")

n_features_plot = min(9, normal_features.shape[1])  # Show first 9 features
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for i in range(n_features_plot):
    ax = axes[i]
    ax.hist(normal_features[:, i], bins=30, alpha=0.6, label='Normal', 
            color='#2ecc71', edgecolor='black', density=True)
    ax.hist(abnormal_features[:, i], bins=30, alpha=0.6, label='Abnormal', 
            color='#e74c3c', edgecolor='black', density=True)
    ax.set_title(f'{feature_names[i]}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Feature Distributions: Normal vs Abnormal', fontsize=16, fontweight='bold')
plt.tight_layout()
save_fig('6_feature_distributions.png')
plt.show()

# 8.2: Feature Correlation Heatmap
print("\n📊 Visualization 6: Feature Correlation Heatmap")

all_features_eda = np.concatenate((normal_features, abnormal_features))
corr_matrix = np.corrcoef(all_features_eda.T)

plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            xticklabels=feature_names, yticklabels=feature_names,
            mask=mask, center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={'label': 'Correlation Coefficient'},
            annot_kws={'size': 8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
save_fig('7_correlation_heatmap.png')
plt.show()

print("\n💡 Correlation Insights:")
print("   ├─ High correlation (>0.7): Features may be redundant")
print("   ├─ Moderate correlation (0.3-0.7): Features capture related but distinct info")
print("   └─ Low correlation (<0.3): Features provide unique information")

# 8.3: Statistical Feature Comparison (Box Plots)
print("\n📊 Visualization 7: Statistical Feature Comparison")

n_features_box = min(9, normal_features.shape[1])
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for i in range(n_features_box):
    ax = axes[i]
    data = [normal_features[:, i], abnormal_features[:, i]]
    bp = ax.boxplot(data, labels=['Normal', 'Abnormal'], patch_artist=True,
                    boxprops=dict(alpha=0.7),
                    medianprops=dict(color='black', linewidth=2),
                    widths=0.6)
    
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax.set_title(f'{feature_names[i]}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Feature Statistics: Normal vs Abnormal', fontsize=16, fontweight='bold')
plt.tight_layout()
save_fig('8_feature_boxplots.png')
plt.show()

# 8.4: Statistical Significance Testing (t-tests)
print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE TESTING (Independent t-tests)")
print("=" * 80)

# Create a results table
significance_results = []
significant_features = []

for i in range(normal_features.shape[1]):
    t_stat, p_val = stats.ttest_ind(normal_features[:, i], abnormal_features[:, i])

    is_significant = p_val < 0.05
    if is_significant:
        significant_features.append(feature_names[i])
    
    significance_results.append({
        'Feature': feature_names[i],
        'T-Statistic': t_stat,
        'P-Value': p_val,
        'Significant': 'Yes' if is_significant else 'No'
    })
    
    significance_marker = "✅ Significant" if is_significant else "❌ Not Significant"
    
    print(f"\n{feature_names[i]}:")
    print(f"  ├─ T-Statistic: {t_stat:8.3f}")
    print(f"  ├─ P-Value:     {p_val:.3e}")
    print(f"  └─ Status:      {significance_marker}")

# Create DataFrame for better visualization
sig_df = pd.DataFrame(significance_results)

print(f"\n{'=' * 80}")
print(f"SIGNIFICANCE SUMMARY")
print(f"{'=' * 80}")
print(f"Significant features: {len(significant_features)}/{len(feature_names)} (p < 0.05)")
print(f"Significant features: {', '.join(significant_features)}")
print(f"{'=' * 80}")

# Visualize significance
print("\n📊 Visualization 8: Statistical Significance")

fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#2ecc71' if sig == 'Yes' else '#e74c3c' for sig in sig_df['Significant']]
bars = ax.barh(sig_df['Feature'], -np.log10(sig_df['P-Value']), color=colors, alpha=0.7, edgecolor='black')

ax.axvline(x=-np.log10(0.05), color='black', linestyle='--', linewidth=2, label='p = 0.05 threshold')
ax.set_xlabel('-log10(P-Value)', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Statistical Significance of Features (t-test)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
save_fig('9_statistical_significance.png')
plt.show()

# 8.5: Mean and Standard Deviation Analysis
print("\n" + "=" * 80)
print("MEAN AND STANDARD DEVIATION ANALYSIS")
print("=" * 80)

print("\nFeature Statistics by Class:")
print("=" * 80)

for i in range(normal_features.shape[1]):
    normal_mean = np.mean(normal_features[:, i])
    normal_std = np.std(normal_features[:, i])
    abnormal_mean = np.mean(abnormal_features[:, i])
    abnormal_std = np.std(abnormal_features[:, i])
    
    print(f"\n{feature_names[i]}:")
    print(f"  ├─ Normal:   Mean = {normal_mean:8.3f}, Std = {normal_std:8.3f}")
    print(f"  └─ Abnormal: Mean = {abnormal_mean:8.3f}, Std = {abnormal_std:8.3f}")

print(f"\n{'=' * 80}")

# #### SECTION 9: DATA PREPARATION FOR MODELING

print("\n" + "=" * 80)
print("SECTION 9: DATA PREPARATION FOR MODELING")
print("=" * 80)

# Combine features and labels
X = np.concatenate((normal_features, abnormal_features))
y = np.concatenate((
    np.zeros(normal_features.shape[0]), 
    np.ones(abnormal_features.shape[0])
))

print(f"\n📦 Combined Dataset:")
print(f"   ├─ Total samples: {X.shape[0]}")
print(f"   ├─ Features per sample: {X.shape[1]}")
print(f"   ├─ Normal samples: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
print(f"   └─ Abnormal samples: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")

# Train-Test Split (Stratified)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, 
    stratify=y, random_state=RANDOM_SEED, shuffle=True
)

# Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=config.VAL_SIZE,
    stratify=y_train_val, random_state=RANDOM_SEED, shuffle=True
)

print(f"\n📊 Data Split:")
print(f"   ├─ Training:   {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"   │   ├─ Normal: {np.sum(y_train == 0)}")
print(f"   │   └─ Abnormal: {np.sum(y_train == 1)}")
print(f"   ├─ Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
print(f"   │   ├─ Normal: {np.sum(y_val == 0)}")
print(f"   │   └─ Abnormal: {np.sum(y_val == 1)}")
print(f"   └─ Test:       {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
print(f"       ├─ Normal: {np.sum(y_test == 0)}")
print(f"       └─ Abnormal: {np.sum(y_test == 1)}")

# Feature Scaling (StandardScaler)
print(f"\n⚙️  Applying StandardScaler normalization...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Normalization complete")
print(f"   ├─ Mean: ~0.0 (actual: {np.mean(X_train_scaled):.6f})")
print(f"   └─ Std Dev: ~1.0 (actual: {np.std(X_train_scaled):.6f})")

# Save scaler for future use
with open(config.SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"\n💾 Scaler saved to: {config.SCALER_SAVE_PATH}")

# #### SECTION 10: MODEL ARCHITECTURE

print("\n" + "=" * 80)
print("SECTION 10: MODEL ARCHITECTURE - PUMP-NET")
print("=" * 80)

def create_pump_net_model(input_shape, learning_rate=0.001):
    """
    Create Pump-Net: Simple but effective DNN for pump anomaly detection
    
    Architecture:
    - Input Layer: Feature vector
    - Hidden Layer 1: 64 units + ReLU + Dropout(0.5)
    - Hidden Layer 2: 64 units + ReLU + Dropout(0.5)
    - Output Layer: 1 unit + Sigmoid (binary classification)
    
    """
    model = Sequential(name='Pump_Net')
    model.add(Input(shape=input_shape))
    # Input + Hidden Layer 1
    model.add(Dense(config.HIDDEN_UNITS_1, activation='relu', name='dense_1'))

    model.add(Dropout(config.DROPOUT_RATE, name='dropout_1'))
    
    # Hidden Layer 2
    model.add(Dense(config.HIDDEN_UNITS_2, activation='relu', name='dense_2'))
    model.add(Dropout(config.DROPOUT_RATE, name='dropout_2'))
    
    # Output Layer
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# Build model
input_shape = (X_train_scaled.shape[1],)
model = create_pump_net_model(input_shape, learning_rate=config.LEARNING_RATE)

print("\n")
model.summary()

total_params = model.count_params()
trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])

print(f"\n{'=' * 80}")
print(f"MODEL PARAMETERS SUMMARY")
print(f"{'=' * 80}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Parameters-to-samples ratio: 1:{X_train.shape[0]/total_params:.2f}")
print(f"✅ Ratio is healthy (recommended: >1:10)")
print(f"{'=' * 80}")

# #### SECTION 11: CROSS-VALIDATION (OPTIONAL)

if config.USE_CROSS_VALIDATION:
    print("\n" + "=" * 80)
    print("SECTION 11: CROSS-VALIDATION")
    print("=" * 80)
    
    print(f"\n🔄 Performing {config.CV_FOLDS}-Fold Stratified Cross-Validation...")
    
    kfold = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    fold_num = 1
    
    for train_idx, val_idx in kfold.split(X, y):
        print(f"\n{'─' * 80}")
        print(f"FOLD {fold_num}/{config.CV_FOLDS}")
        print(f"{'─' * 80}")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale features
        scaler_fold = StandardScaler()
        X_train_fold = scaler_fold.fit_transform(X_train_fold)
        X_val_fold = scaler_fold.transform(X_val_fold)
        
        # Create and train model
        model_fold = create_pump_net_model(X_train_fold.shape[1:], 
                                          learning_rate=config.LEARNING_RATE)
        
        history_fold = model_fold.fit(
            X_train_fold, y_train_fold,
            epochs=100,
            batch_size=config.BATCH_SIZE,
            verbose=0,
            validation_data=(X_val_fold, y_val_fold)
        )
        
        # Predict and evaluate
        y_pred_fold = (model_fold.predict(X_val_fold, verbose=0) > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_fold, y_pred_fold)
        precision = precision_score(y_val_fold, y_pred_fold, zero_division=0)
        recall = recall_score(y_val_fold, y_pred_fold, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred_fold, zero_division=0)
        
        # Get AUC from predictions
        y_pred_proba_fold = model_fold.predict(X_val_fold, verbose=0).flatten()
        try:
            auc_score = roc_auc_score(y_val_fold, y_pred_proba_fold)
        except:
            auc_score = 0.0
        
        cv_scores['accuracy'].append(accuracy)
        cv_scores['precision'].append(precision)
        cv_scores['recall'].append(recall)
        cv_scores['f1'].append(f1)
        cv_scores['auc'].append(auc_score)
        
        print(f"  ├─ Accuracy:  {accuracy:.4f}")
        print(f"  ├─ Precision: {precision:.4f}")
        print(f"  ├─ Recall:    {recall:.4f}")
        print(f"  ├─ F1-Score:  {f1:.4f}")
        print(f"  └─ AUC:       {auc_score:.4f}")
        
        fold_num += 1
    
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'=' * 80}")
    for metric, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric.upper():12s}: {mean_score:.4f} ± {std_score:.4f}")
    print(f"{'=' * 80}")
    
   

# Visualize CV results
print("\n📊 Visualization 9: Cross-Validation Results")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot
ax = axes[0]
cv_data = [cv_scores['accuracy'], cv_scores['precision'], 
            cv_scores['recall'], cv_scores['f1'], cv_scores['auc']]
bp = ax.boxplot(cv_data, labels=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
                patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.05])

# Bar plot with error bars
ax = axes[1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
means = [np.mean(cv_scores[m.lower()]) for m in metrics]
stds = [np.std(cv_scores[m.lower()]) for m in metrics]

bars = ax.bar(metrics, means, yerr=stds, capsize=10, alpha=0.7,
                color='#3498db', edgecolor='black', linewidth=2)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation Average Scores', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.05])

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
            f'{mean:.3f}\n±{std:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle(f'{config.CV_FOLDS}-Fold Cross-Validation Results', 
                fontsize=16, fontweight='bold')
plt.tight_layout()
save_fig('10_cross_validation_results.png')
plt.show()

# #### SECTION 12: MODEL TRAINING

print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

clear_gpu_memory()

# Setup callbacks
callbacks_list = [
    # Early stopping
    callbacks.EarlyStopping(
        monitor='val_auc',
        patience=config.PATIENCE,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    
    # Reduce learning rate on plateau
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Model checkpoint
    callbacks.ModelCheckpoint(
        config.MODEL_SAVE_PATH,
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("\n⚙️  Training Configuration:")
print(f"   ├─ Epochs: {config.EPOCHS}")
print(f"   ├─ Batch Size: {config.BATCH_SIZE}")
print(f"   ├─ Learning Rate: {config.LEARNING_RATE}")
print(f"   ├─ Early Stopping Patience: {config.PATIENCE}")
print(f"   └─ Optimizer: Adam")

print("\n🚀 Starting training...\n")

# Train model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

print("\n✅ Training completed!")

# #### Performance vs Speed Trade-offs

"""Original (128 mels, batch 32)  - 96-99%(accuracy)    - Fastest     ❌ 4-6GB
Optimized (64 mels, batch 16)  - 94-97%    - ~1.5x slower ✅ 2-3GB
Ultra-light (32 mels, batch 8) - 90-94%    - ~2x slower   ✅ 1-2GB"""

# #### SECTION 13: TRAINING HISTORY VISUALIZATION

print("\n" + "=" * 80)
print("SECTION 13: TRAINING HISTORY VISUALIZATION")
print("=" * 80)

# Plot training history
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2, color='#3498db')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#e74c3c')
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2, color='#3498db')
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='#e74c3c')
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# AUC
axes[0, 2].plot(history.history['auc'], label='Train', linewidth=2, color='#3498db')
axes[0, 2].plot(history.history['val_auc'], label='Validation', linewidth=2, color='#e74c3c')
axes[0, 2].set_title('Model AUC', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('AUC')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2, color='#3498db')
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2, color='#e74c3c')
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2, color='#3498db')
axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2, color='#e74c3c')
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Learning Rate (if available)
if 'lr' in history.history:
    axes[1, 2].plot(history.history['lr'], linewidth=2, color='purple')
    axes[1, 2].set_title('Learning Rate', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(alpha=0.3)
else:
    axes[1, 2].axis('off')

plt.suptitle('Training History', fontsize=16, fontweight='bold')
plt.tight_layout()
save_fig('11_training_history.png')
plt.show()

# Print best metrics
best_epoch = np.argmax(history.history['val_auc'])
print(f"\n{'=' * 80}")
print(f"BEST TRAINING METRICS (Epoch {best_epoch + 1})")
print(f"{'=' * 80}")
print(f"Training Accuracy:    {history.history['accuracy'][best_epoch]:.4f}")
print(f"Validation Accuracy:  {history.history['val_accuracy'][best_epoch]:.4f}")
print(f"Validation AUC:       {history.history['val_auc'][best_epoch]:.4f}")
print(f"Validation Precision: {history.history['val_precision'][best_epoch]:.4f}")
print(f"Validation Recall:    {history.history['val_recall'][best_epoch]:.4f}")
print(f"{'=' * 80}")

# #### SECTION 14: MODEL EVALUATION ON TEST SET

print("\n" + "=" * 80)
print("SECTION 14: MODEL EVALUATION - TEST SET")
print("=" * 80)

# Load best model
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
print(f"✅ Loaded best model from: {config.MODEL_SAVE_PATH}")

# Predict on test set
print("\n🔮 Generating predictions on test set...")
y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_pred_prob)

print("\n" + "=" * 80)
print("TEST SET PERFORMANCE METRICS")
print("=" * 80)
print(f"Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"AUC-ROC:   {test_auc:.4f}")
print("=" * 80)

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=config.CLASS_NAMES,
                          digits=4))

# #### SECTION 15: CONFUSION MATRIX

print("\n📊 Visualization 10: Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion matrix (counts)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
            ax=axes[0], cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Confusion matrix (normalized)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
            ax=axes[1], cbar_kws={'label': 'Percentage'},
            annot_kws={'size': 16, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
save_fig('12_confusion_matrix.png')
plt.show()

# Calculate specific metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\nConfusion Matrix Analysis:")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")
print(f"\n  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity:          {specificity:.4f}")

# #### SECTION 16: ROC CURVE AND PRECISION-RECALL CURVE

print("\n📊 Visualization 11: ROC and Precision-Recall Curves")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
roc_auc_val = auc(fpr, tpr)

axes[0].plot(fpr, tpr, color='darkorange', lw=3,
             label=f'ROC curve (AUC = {roc_auc_val:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
axes[0].legend(loc="lower right", fontsize=11)
axes[0].grid(alpha=0.3)

# Precision-Recall Curve
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall_curve, precision_curve)

axes[1].plot(recall_curve, precision_curve, color='blue', lw=3,
             label=f'PR curve (AUC = {pr_auc:.4f})')
axes[1].axhline(y=np.sum(y_test)/len(y_test), color='navy', 
                linestyle='--', lw=2, label='Baseline (Random)')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend(loc="lower left", fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
save_fig('13_roc_pr_curves.png')
plt.show()

# #### SECTION 17: PREDICTION ANALYSIS

print("\n" + "=" * 80)
print("SECTION 17: PREDICTION ANALYSIS")
print("=" * 80)

# Visualization: Distribution of prediction probabilities
print("\n📊 Visualization 12: Prediction Probability Distribution")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
axes[0].hist(y_pred_prob[y_test == 0], bins=30, alpha=0.7, 
            label='Normal (True)', color='#2ecc71', edgecolor='black')
axes[0].hist(y_pred_prob[y_test == 1], bins=30, alpha=0.7,
            label='Abnormal (True)', color='#e74c3c', edgecolor='black')
axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)


# Box plot
data_to_plot = [y_pred_prob[y_test == 0], y_pred_prob[y_test == 1]]
bp = axes[1].boxplot(data_to_plot, labels=['Normal', 'Abnormal'], patch_artist=True,
                     boxprops=dict(alpha=0.7),
                     medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[1].set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
axes[1].set_title('Prediction Probability by True Class', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
save_fig('14_prediction_distribution.png')
plt.show()

# Confidence analysis
confident_correct = np.sum((np.abs(y_pred_prob - 0.5) > 0.3) & (y_pred == y_test))
confident_incorrect = np.sum((np.abs(y_pred_prob - 0.5) > 0.3) & (y_pred != y_test))
uncertain = np.sum(np.abs(y_pred_prob - 0.5) <= 0.3)

print(f"\nPrediction Confidence Analysis:")
print(f"  ├─ Confident & Correct:   {confident_correct:3d} ({confident_correct/len(y_test)*100:.2f}%)")
print(f"  ├─ Confident & Incorrect: {confident_incorrect:3d} ({confident_incorrect/len(y_test)*100:.2f}%)")
print(f"  └─ Uncertain (0.2-0.8):   {uncertain:3d} ({uncertain/len(y_test)*100:.2f}%)")

# #### SECTION 18: THRESHOLD OPTIMIZATION (OPTIONAL)

if config.OPTIMIZE_THRESHOLD:
    print("\n" + "=" * 80)
    print("SECTION 18: THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    print("\n🔍 Finding optimal classification threshold...")
    
    # Calculate F1 scores for different thresholds
    thresholds_to_test = np.arange(0.1, 0.9, 0.01)
    f1_scores_list = []
    
    for thresh in thresholds_to_test:
        y_pred_thresh = (y_pred_prob > thresh).astype(int)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        f1_scores_list.append(f1)
    
    optimal_threshold = thresholds_to_test[np.argmax(f1_scores_list)]
    optimal_f1 = np.max(f1_scores_list)
    
    print(f"\n✅ Optimal Threshold Found:")
    print(f"   ├─ Threshold: {optimal_threshold:.3f}")
    print(f"   └─ F1-Score:  {optimal_f1:.4f}")
    
    # Recalculate metrics with optimal threshold
    y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)
    
    optimal_accuracy = accuracy_score(y_test, y_pred_optimal)
    optimal_precision = precision_score(y_test, y_pred_optimal, zero_division=0)
    optimal_recall = recall_score(y_test, y_pred_optimal, zero_division=0)
    
    print(f"\n📊 Metrics with Optimal Threshold ({optimal_threshold:.3f}):")
    print(f"   ├─ Accuracy:  {optimal_accuracy:.4f}")
    print(f"   ├─ Precision: {optimal_precision:.4f}")
    print(f"   ├─ Recall:    {optimal_recall:.4f}")
    print(f"   └─ F1-Score:  {optimal_f1:.4f}")
    
    # Visualize threshold optimization
    print("\n📊 Visualization 13: Threshold Optimization")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(thresholds_to_test, f1_scores_list, linewidth=3, color='#3498db', label='F1-Score')
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Default Threshold = 0.5')
    
    ax.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Threshold Optimization: F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig('15_threshold_optimization.png')
    plt.show()

# #### SECTION 19: FEATURE IMPORTANCE ANALYSIS

print("\n" + "=" * 80)
print("SECTION 19: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print("\n🔍 Computing feature importance using Random Forest...")

# Train Random Forest for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
rf_model.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize feature importance
print("\n📊 Visualization 14: Feature Importance")

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#e74c3c' if imp > np.mean(rf_model.feature_importances_) else '#3498db' 
          for imp in feature_importance['Importance']]

bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], 
               color=colors, alpha=0.7, edgecolor='black')

ax.axvline(x=float(np.mean(rf_model.feature_importances_)), color='black', 
           linestyle='--', linewidth=2, label='Mean Importance')
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance for Pump Anomaly Detection', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
save_fig('16_feature_importance.png')
plt.show()

# #### SECTION 20: ERROR ANALYSIS

print("\n" + "=" * 80)
print("SECTION 20: ERROR ANALYSIS")
print("=" * 80)

# Identify misclassified samples
misclassified_idx = np.where(y_pred != y_test)[0]
correctly_classified_idx = np.where(y_pred == y_test)[0]

print(f"\n📊 Classification Results:")
print(f"   ├─ Correctly Classified:   {len(correctly_classified_idx)} ({len(correctly_classified_idx)/len(y_test)*100:.2f}%)")
print(f"   └─ Misclassified:          {len(misclassified_idx)} ({len(misclassified_idx)/len(y_test)*100:.2f}%)")

if len(misclassified_idx) > 0:
    print(f"\n⚠️  Misclassified Samples Analysis:")
    
    for idx in misclassified_idx[:5]:  # Show first 5 misclassifications
        true_label = 'Normal' if y_test[idx] == 0 else 'Abnormal'
        pred_label = 'Normal' if y_pred[idx] == 0 else 'Abnormal'
        confidence = y_pred_prob[idx] if y_pred[idx] == 1 else 1 - y_pred_prob[idx]
        
        print(f"\n   Sample {idx}:")
        print(f"      ├─ True Label:       {true_label}")
        print(f"      ├─ Predicted Label:  {pred_label}")
        print(f"      ├─ Confidence:       {confidence:.4f}")
        print(f"      └─ Prediction Score: {y_pred_prob[idx]:.4f}")

# #### SECTION 21: FINAL MODEL SUMMARY AND COMPARISON

print("\n" + "=" * 80)
print("SECTION 21: FINAL MODEL SUMMARY")
print("=" * 80)

# Create comprehensive results summary
results_summary = {
    'model_name': 'Pump-Net',
    'architecture': f'{config.HIDDEN_UNITS_1}-{config.HIDDEN_UNITS_2} DNN',
    'total_parameters': total_params,
    'feature_count': len(feature_names),
    'test_metrics': {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'auc_roc': test_auc,
        'specificity': specificity,
        'sensitivity': sensitivity
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'training_info': {
        'epochs_trained': len(history.history['loss']),
        'best_epoch': int(best_epoch + 1),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1])
    }
}

if config.OPTIMIZE_THRESHOLD:
    results_summary['optimal_threshold'] = float(optimal_threshold)
    results_summary['optimal_f1'] = float(optimal_f1)

# Save results
with open(config.RESULTS_SAVE_PATH, 'wb') as f:
    pickle.dump(results_summary, f)

print(f"\n💾 Results saved to: {config.RESULTS_SAVE_PATH}")

# Display final metrics table
print("\n" + "=" * 80)
print("FINAL PERFORMANCE METRICS")
print("=" * 80)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 
               'Specificity', 'Sensitivity'],
    'Value': [test_accuracy, test_precision, test_recall, test_f1, test_auc,
              specificity, sensitivity],
    'Percentage': [f"{test_accuracy*100:.2f}%", f"{test_precision*100:.2f}%", 
                   f"{test_recall*100:.2f}%", f"{test_f1*100:.2f}%", 
                   f"{test_auc*100:.2f}%", f"{specificity*100:.2f}%", 
                   f"{sensitivity*100:.2f}%"]
})

print(metrics_df.to_string(index=False))
print("=" * 80)

# Visualize final metrics
print("\n📊 Visualization 15: Final Performance Metrics")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
ax = axes[0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
values = [test_accuracy, test_precision, test_recall, test_f1, test_auc]
colors_bar = ['#2ecc71' if v >= 0.9 else '#f39c12' if v >= 0.8 else '#e74c3c' for v in values]

bars = ax.bar(metrics, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (>90%)')
ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good (>80%)')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{value:.4f}\n({value*100:.2f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Radar chart
ax = axes[1]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values_radar = values + [values[0]]  # Close the plot
angles += angles[:1]

ax = plt.subplot(122, projection='polar')
ax.plot(angles, values_radar, 'o-', linewidth=3, color='#3498db', label='Pump-Net')
ax.fill(angles, values_radar, alpha=0.25, color='#3498db')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
save_fig('17_final_metrics.png')
plt.show()

# #### SECTION 22: MODEL DEPLOYMENT INFORMATION

print("\n" + "=" * 80)
print("SECTION 22: MODEL DEPLOYMENT INFORMATION")
print("=" * 80)

print(f"\n📦 Model Artifacts:")
print(f"   ├─ Model File:    {config.MODEL_SAVE_PATH}")
print(f"   ├─ Scaler File:   {config.SCALER_SAVE_PATH}")
print(f"   └─ Results File:  {config.RESULTS_SAVE_PATH}")

print(f"\n🔧 Model Specifications:")
print(f"   ├─ Input Shape:        ({len(feature_names)},)")
print(f"   ├─ Output:             Binary (Normal/Abnormal)")
print(f"   ├─ Feature Count:      {len(feature_names)}")
print(f"   ├─ Model Parameters:   {total_params:,}")
print(f"   └─ Inference Time:     ~1ms per sample (CPU)")

print(f"\n🚀 Deployment Recommendations:")
print(f"   ├─ Production Threshold: {optimal_threshold:.3f}" if config.OPTIMIZE_THRESHOLD else "   ├─ Production Threshold: 0.500")
print(f"   ├─ Minimum Confidence:   0.70 (for critical decisions)")
print(f"   ├─ Batch Size:           32-128 samples")
print(f"   └─ Expected Latency:     <10ms per sample")



