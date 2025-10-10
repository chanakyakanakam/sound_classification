"""
Path: /home/chanakya/sound_classification/ui/streamlit_app.py
Streamlit UI for Pump-Net
Interactive web interface for pump anomaly detection
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Pump-Net | Anomaly Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .alert-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .alert-abnormal {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_audio(audio_file):
    """Send audio file to API for prediction"""
    try:
        files = {'file': audio_file}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, str(e)

def create_gauge_chart(probability, title):
    """Create gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#d4edda'},
                {'range': [50, 75], 'color': '#fff3cd'},
                {'range': [75, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_probability_bar_chart(prob_normal, prob_abnormal):
    """Create bar chart for probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Normal', 'Abnormal'],
            y=[prob_normal * 100, prob_abnormal * 100],
            marker_color=['#28a745', '#dc3545'],
            text=[f'{prob_normal*100:.2f}%', f'{prob_abnormal*100:.2f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        yaxis_title='Probability (%)',
        height=400,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    return fig

def visualize_waveform(audio_file):
    """Visualize audio waveform"""
    # Load audio
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color='#1f77b4', alpha=0.8)
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3)
    
    return fig

def visualize_spectrogram(audio_file):
    """Visualize mel-spectrogram"""
    # Load audio
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                    sr=sr, ax=ax, cmap='viridis')
    ax.set_title('Mel-Spectrogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔧 Pump-Net</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Industrial Pump Anomaly Detection System</p>', unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Pump-Net", use_column_width=True)
        st.markdown("---")
        
        st.subheader("System Status")
        if api_status:
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            st.info("Please start the API server:\n```bash\npython api/main.py\n```")
        
        st.markdown("---")
        
        st.subheader("About")
        st.info(
            "**Pump-Net** uses advanced machine learning to detect anomalies "
            "in industrial pump sounds. Upload a .wav file to analyze."
        )
        
        st.markdown("---")
        
        st.subheader("Model Info")
        st.metric("Accuracy", "98.07%")
        st.metric("Precision", "100.00%")
        st.metric("Recall", "91.23%")
        st.metric("F1-Score", "95.41%")
    
    # Main content
    if not api_status:
        st.error("⚠️ Cannot connect to API server. Please ensure the API is running.")
        st.stop()
    
    # File uploader
    st.markdown("### 📂 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose a .wav file",
        type=['wav'],
        help="Upload an audio file of pump sound for analysis"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("Format", "WAV")
        
        st.markdown("---")
        
        # Audio player
        st.markdown("### 🎵 Audio Preview")
        st.audio(uploaded_file, format='audio/wav')
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📊 Audio Visualization")
        
        # Save to temp file for visualization
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            with st.spinner("Generating waveform..."):
                waveform_fig = visualize_waveform(tmp_path)
                st.pyplot(waveform_fig)
                plt.close()
        
        with viz_col2:
            with st.spinner("Generating spectrogram..."):
                spectrogram_fig = visualize_spectrogram(tmp_path)
                st.pyplot(spectrogram_fig)
                plt.close()
        
        st.markdown("---")
        
        # Predict button
        st.markdown("### 🔮 Analysis")
        
        if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing audio... Please wait..."):
                uploaded_file.seek(0)
                result, error = predict_audio(uploaded_file)
                
                # Fix: Check if result is None before accessing
                if error or result is None:
                    st.error(f"❌ Prediction failed: {error or 'Unknown error'}")
                else:
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📈 Analysis Results")
                    
                    # Verdict
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    if prediction == 'Normal':
                        st.markdown(
                            f'<div class="alert-normal">'
                            f'<h2>✅ {prediction} Operation</h2>'
                            f'<p style="font-size: 1.2rem;">No anomalies detected. '
                            f'Pump is operating normally.</p>'
                            f'<p><strong>Confidence:</strong> {confidence*100:.2f}%</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="alert-abnormal">'
                            f'<h2>⚠️ {prediction} Detected</h2>'
                            f'<p style="font-size: 1.2rem;">Anomaly detected! '
                            f'Pump requires inspection.</p>'
                            f'<p><strong>Confidence:</strong> {confidence*100:.2f}%</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("---")
                    
                    # Probability gauges
                    st.markdown("### 📊 Probability Analysis")
                    
                    gauge_col1, gauge_col2 = st.columns(2)
                    
                    with gauge_col1:
                        normal_gauge = create_gauge_chart(
                            result['probability_normal'],
                            "Normal Probability"
                        )
                        st.plotly_chart(normal_gauge, use_container_width=True)
                    
                    with gauge_col2:
                        abnormal_gauge = create_gauge_chart(
                            result['probability_abnormal'],
                            "Abnormal Probability"
                        )
                        st.plotly_chart(abnormal_gauge, use_container_width=True)
                    
                    # Bar chart
                    st.markdown("### 📊 Comparison Chart")
                    bar_chart = create_probability_bar_chart(
                        result['probability_normal'],
                        result['probability_abnormal']
                    )
                    st.plotly_chart(bar_chart, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Detailed metrics
                    st.markdown("### 📋 Detailed Metrics")
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric(
                            "Prediction",
                            result['prediction'],
                            delta="Normal" if result['prediction'] == 'Normal' else "Anomaly"
                        )
                    
                    with metrics_col2:
                        st.metric(
                            "Confidence Level",
                            f"{result['confidence']*100:.2f}%",
                            delta="High" if result['is_confident'] else "Low"
                        )
                    
                    with metrics_col3:
                        st.metric(
                            "Threshold Used",
                            f"{result['threshold_used']*100:.0f}%"
                        )
                    
                    # Raw results (expandable)
                    with st.expander("🔍 View Raw Results"):
                        st.json(result)
                    
                    # Recommendations
                    st.markdown("---")
                    st.markdown("### 💡 Recommendations")
                    
                    if prediction == 'Normal':
                        st.success(
                            "✅ **Continue normal operations**\n\n"
                            "- Monitor regularly for any changes\n"
                            "- Maintain scheduled maintenance\n"
                            "- Keep logs for trend analysis"
                        )
                    else:
                        st.warning(
                            "⚠️ **Immediate action required**\n\n"
                            "- Schedule inspection immediately\n"
                            "- Check for unusual vibrations or sounds\n"
                            "- Review maintenance records\n"
                            "- Consider backup pump activation"
                        )
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
    
    else:
        # Instructions when no file uploaded
        st.info(
            "👆 **Get Started:** Upload a .wav audio file of pump sound to begin analysis.\n\n"
            "The system will analyze the audio and detect any anomalies in real-time."
        )
        
        # Example section
        st.markdown("---")
        st.markdown("### 📚 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                "#### 1️⃣ Upload Audio\n"
                "Upload a .wav file of pump sound recording"
            )
        
        with col2:
            st.markdown(
                "#### 2️⃣ AI Analysis\n"
                "Advanced ML model analyzes acoustic features"
            )
        
        with col3:
            st.markdown(
                "#### 3️⃣ Get Results\n"
                "Receive instant diagnosis with confidence scores"
            )

if __name__ == "__main__":
    main()
