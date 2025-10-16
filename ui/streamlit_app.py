"""
Path: /home/chanakya/sound_classification/ui/streamlit_app.py
Streamlit UI for Pump-Net with Similarity-Based Explainability
Interactive web interface for pump anomaly detection
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import tempfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json

# Page configuration
st.set_page_config(
    page_title="Anomaly Pump Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
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
    .stMarkdown h3 {
        font-size: 1.3rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
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

def create_2d_scatter_plot(viz_data, user_projection):
    """Create 2D scatter plot showing training data and user's point"""
    if 'error' in viz_data:
        return None
    
    # Prepare data
    df = pd.DataFrame({
        'x': viz_data['x'],
        'y': viz_data['y'],
        'label': ['Normal' if l == 0 else 'Abnormal' for l in viz_data['labels']]
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add training data points
    for label, color in [('Normal', '#28a745'), ('Abnormal', '#dc3545')]:
        mask = df['label'] == label
        fig.add_trace(go.Scatter(
            x=df[mask]['x'],
            y=df[mask]['y'],
            mode='markers',
            name=f'{label} (Training)',
            marker=dict(
                size=8,
                color=color,
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate=f'<b>{label}</b><br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>'
        ))
    
    # Add user's point
    fig.add_trace(go.Scatter(
        x=[user_projection[0]],
        y=[user_projection[1]],
        mode='markers',
        name='Your Pump',
        marker=dict(
            size=20,
            color='#FFD700',
            symbol='star',
            line=dict(width=2, color='#FF8C00')
        ),
        hovertemplate='<b>YOUR PUMP</b><br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Feature Space Visualization ({viz_data['projection_method'].upper()})",
        xaxis_title=f"{viz_data['projection_method'].upper()} Component 1",
        yaxis_title=f"{viz_data['projection_method'].upper()} Component 2",
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def load_training_report():
    """Load training report from JSON file"""
    report_path = Path("/home/chanakya/sound_classification/models/training_report.json")
    
    try:
        if report_path.exists():
            with open(report_path, 'r') as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading training report: {e}")
        return None

def create_cluster_distance_chart(cluster_distances):
    """Create bar chart for cluster distances"""
    if 'error' in cluster_distances:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Distance to Normal', 'Distance to Abnormal'],
        y=[cluster_distances['distance_to_normal'], cluster_distances['distance_to_abnormal']],
        marker_color=['#28a745', '#dc3545'],
        text=[f"{cluster_distances['distance_to_normal']:.2f}", 
              f"{cluster_distances['distance_to_abnormal']:.2f}"],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Distance to Cluster Centers',
        yaxis_title='Euclidean Distance',
        height=400,
        showlegend=False
    )
    
    return fig

def create_feature_importance_chart(feature_importance, top_n=10):
    """Create horizontal bar chart for feature importance"""
    if not feature_importance:
        return None
    
    # Take top N features
    top_features = feature_importance[:top_n]
    
    df = pd.DataFrame(top_features)
    
    # Color based on deviation level
    colors = ['#dc3545' if d > 2.0 else '#ff9800' if d > 1.0 else '#2196F3' 
              for d in df['deviation']]
    
    fig = go.Figure(go.Bar(
        x=df['deviation'],
        y=df['feature_name'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{d:.2f}" for d in df['deviation']],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Deviation: %{x:.2f}<br>Value: %{customdata[0]:.3f}<br>Normal Mean: %{customdata[1]:.3f}<extra></extra>',
        customdata=df[['value', 'normal_mean']].values
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features (Deviation from Normal)',
        xaxis_title='Standard Deviations from Normal',
        yaxis_title='',
        height=400,
        showlegend=False
    )
    
    return fig

def visualize_waveform(audio_file):
    """Visualize audio waveform"""
    audio, sr = librosa.load(audio_file, sr=None)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color='#1f77b4', alpha=0.8)
    ax.set_title('Audio Waveform', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(alpha=0.3)
    
    return fig

def visualize_spectrogram(audio_file):
    """Visualize mel-spectrogram"""
    audio, sr = librosa.load(audio_file, sr=None)
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                    sr=sr, ax=ax, cmap='viridis')
    ax.set_title('Mel-Spectrogram', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔧 Pump-Net</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Industrial Pump Anomaly Detection with AI Explainability</p>', unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <style>
            [data-testid="stSidebar"] h3 {
                font-size: 1.1rem !important;
            }
            [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
                font-size: 0.85rem !important;
            }
            [data-testid="stSidebar"] [data-testid="stMetricValue"] {
                font-size: 1.1rem !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.subheader("System Status")
        if api_status:
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            st.info("Please start the API server:\n```bash\npython -m api.main\n```")
        
        st.markdown("---")
        
        # Load training report
        training_report = load_training_report()
        
        if training_report:
            # ============================================================
            # DATA SPLIT INFORMATION
            # ============================================================
            st.subheader("📊 Data Split")
            
            data_split = training_report.get('data_split', {})
            
            # Training split
            training_info = data_split.get('training', {})
            with st.expander("🟢 Training Set", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Total", 
                        training_info.get('total_samples', 'N/A')
                    )
                    st.metric(
                        "Normal", 
                        training_info.get('class_distribution', {}).get('normal', 'N/A')
                    )
                with col2:
                    st.metric(
                        "Percentage", 
                        f"{training_info.get('percentage', 0):.1f}%"
                    )
                    st.metric(
                        "Abnormal", 
                        training_info.get('class_distribution', {}).get('abnormal', 'N/A')
                    )
            
            # Validation split
            validation_info = data_split.get('validation', {})
            with st.expander("🟡 Validation Set"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Total", 
                        validation_info.get('total_samples', 'N/A')
                    )
                    st.metric(
                        "Normal", 
                        validation_info.get('class_distribution', {}).get('normal', 'N/A')
                    )
                with col2:
                    st.metric(
                        "Percentage", 
                        f"{validation_info.get('percentage', 0):.1f}%"
                    )
                    st.metric(
                        "Abnormal", 
                        validation_info.get('class_distribution', {}).get('abnormal', 'N/A')
                    )
            
            # Testing split
            testing_info = data_split.get('testing', {})
            with st.expander("🔵 Test Set"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Total", 
                        testing_info.get('total_samples', 'N/A')
                    )
                    st.metric(
                        "Normal", 
                        testing_info.get('class_distribution', {}).get('normal', 'N/A')
                    )
                with col2:
                    st.metric(
                        "Percentage", 
                        f"{testing_info.get('percentage', 0):.1f}%"
                    )
                    st.metric(
                        "Abnormal", 
                        testing_info.get('class_distribution', {}).get('abnormal', 'N/A')
                    )
            
            st.markdown("---")
            
            # ============================================================
            # MODEL PERFORMANCE (From JSON)
            # ============================================================
            st.subheader("🎯 Model Performance")

            test_metrics = training_report.get('test_performance', {}).get('metrics', {})

            # Display metrics from JSON
            accuracy = test_metrics.get('accuracy', {}).get('percentage', '98.07%')
            precision = test_metrics.get('precision', {}).get('percentage', '100.00%')
            recall = test_metrics.get('recall', {}).get('percentage', '91.23%')
            f1_score = test_metrics.get('f1_score', {}).get('percentage', '95.41%')
            
            with st.expander("📊 Performance Metrics", expanded=True):
                st.metric("Accuracy", accuracy)
                st.metric("Precision", precision)
                st.metric("Recall", recall)
                st.metric("F1-Score", f1_score)

            
            # Confusion Matrix Summary
            with st.expander("📊 Confusion Matrix"):
                cm = training_report.get('test_performance', {}).get('confusion_matrix', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("True Neg", cm.get('true_negatives', 'N/A'))
                    st.metric("False Pos", cm.get('false_positives', 'N/A'))
                with col2:
                    st.metric("False Neg", cm.get('false_negatives', 'N/A'))
                    st.metric("True Pos", cm.get('true_positives', 'N/A'))
            
            # Training Info
            with st.expander("ℹ️ Training Info"):
                training_history = training_report.get('training_history', {})
                metadata = training_report.get('metadata', {})
                
                st.text(f"Trained: {metadata.get('training_timestamp', 'Unknown')}")
                st.text(f"Model: {metadata.get('model_name', 'Pump-Net')}")
        
        else:
            # Fallback if JSON not available
            st.subheader("📊 Data Split")
            st.warning("Training report not found. Please run training first.")
            
            st.markdown("---")
            
            st.subheader("🎯 Model Performance")
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
        
        # Audio player
        st.markdown("### 🎵 Audio Preview")
        st.audio(uploaded_file, format='audio/wav')
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Predict button
        st.markdown("### 🔮 Analysis")
        
        if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing audio with AI explainability... Please wait..."):
                uploaded_file.seek(0)
                result, error = predict_audio(uploaded_file)
                
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
                    
                    # Check if explainability is available
                    explainability = result.get('explainability', {})
                    
                    if explainability.get('available', False):
                        # ========================================================
                        # AI EXPLAINABILITY DASHBOARD
                        # ========================================================
                        st.markdown("### 🤖 AI Explainability: Why This Decision?")
                        st.markdown("*Understanding the AI's reasoning through similarity analysis*")
                        
                        # Row 1: Feature Space Visualization
                        st.markdown("#### 📊 Where Does Your Pump Sound Sit?")
                        st.markdown("This visualization shows how your pump sound compares to all training examples in a 2D feature space.")
                        
                        viz_data = explainability.get('visualization_data', {})
                        user_projection = explainability.get('projection_2d', (0, 0))
                        
                        scatter_plot = create_2d_scatter_plot(viz_data, user_projection)
                        if scatter_plot:
                            st.plotly_chart(scatter_plot, use_container_width=True)
                            
                            # Interpretation
                            cluster_distances = explainability.get('cluster_distances', {})
                            if 'closer_to' in cluster_distances:
                                closer_to = cluster_distances['closer_to']
                                ratio = cluster_distances.get('distance_ratio', 1.0)
                                
                                if closer_to == 'Abnormal':
                                    st.warning(f"⚠️ **Your pump is {ratio:.1f}x closer to the abnormal cluster than to normal samples**")
                                else:
                                    st.success(f"✅ **Your pump is closer to the normal cluster** (ratio: {1/ratio if ratio > 0 else 'inf'}x)")
                        
                        st.markdown("---")
                        
                        # Row 2: Similar Examples
                        st.markdown("#### 🔍 Most Similar Pump Sounds from Training Data")
                        st.markdown("The AI compared your pump to all training examples and found these most similar sounds:")
                        
                        similar_examples = explainability.get('similar_examples', {})
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### 🟢 Similar Normal Examples")
                            similar_normal = similar_examples.get('similar_normal', [])
                            if similar_normal:
                                for i, example in enumerate(similar_normal, 1):
                                    similarity_pct = example['similarity'] * 100
                                    st.info(f"**#{i}**: {similarity_pct:.1f}% similar (distance: {example['distance']:.3f})")
                            else:
                                st.warning("No similar normal examples found")
                        
                        with col2:
                            st.markdown("##### 🔴 Similar Abnormal Examples")
                            similar_abnormal = similar_examples.get('similar_abnormal', [])
                            if similar_abnormal:
                                for i, example in enumerate(similar_abnormal, 1):
                                    similarity_pct = example['similarity'] * 100
                                    st.error(f"**#{i}**: {similarity_pct:.1f}% similar (distance: {example['distance']:.3f})")
                            else:
                                st.warning("No similar abnormal examples found")
                        
                        # Overall verdict
                        closest = similar_examples.get('closest_overall', 'Unknown')
                        if closest == 'Abnormal':
                            st.error("🎯 **Verdict**: Your pump sounds most similar to ABNORMAL training examples")
                        else:
                            st.success("🎯 **Verdict**: Your pump sounds most similar to NORMAL training examples")
                        
                        st.markdown("---")
                        
                        # Row 3: Cluster Distances & Feature Importance
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📏 Distance to Cluster Centers")
                            cluster_chart = create_cluster_distance_chart(cluster_distances)
                            if cluster_chart:
                                st.plotly_chart(cluster_chart, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### ⚡ Top Contributing Features")
                            feature_importance = explainability.get('feature_importance', [])
                            importance_chart = create_feature_importance_chart(feature_importance, top_n=10)
                            if importance_chart:
                                st.plotly_chart(importance_chart, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Interpretation Guide
                        st.markdown("### 📖 How to Interpret These Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **🎯 Feature Space Plot**
                            - 🟢 Green dots = Normal training samples
                            - 🔴 Red dots = Abnormal training samples
                            - ⭐ Yellow star = YOUR pump
                            - **Clusters show the AI learned distinct patterns**
                            - Your position shows which pattern you match
                            """)
                            
                            st.markdown("""
                            **📏 Distance Metrics**
                            - Shows how far your pump is from each cluster
                            - Lower distance = More similar
                            - If "Distance to Abnormal" is much lower → Likely abnormal
                            """)
                        
                        with col2:
                            st.markdown("""
                            **🔍 Similar Examples**
                            - AI found the 3 most similar sounds from training
                            - Higher similarity % = More alike
                            - Check which category (Normal/Abnormal) dominates
                            - This is the AI's "memory" of past examples
                            """)
                            
                            st.markdown("""
                            **⚡ Feature Importance**
                            - Shows which acoustic features differ most from normal
                            - 🔴 Red bars = Critical deviations (>2σ)
                            - 🟠 Orange bars = Significant deviations (1-2σ)
                            - 🔵 Blue bars = Minor deviations (<1σ)
                            """)
                    
                    else:
                        st.warning("⚠️ Explainability data not available. The model may need to be retrained with explainability support.")
                    
                    st.markdown("---")
                    
                    # Detailed metrics
                    st.markdown("### 📋 Detailed Metrics")
                    
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
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
                            "Normal Probability",
                            f"{result['probability_normal']*100:.2f}%"
                        )

                    with metrics_col4:
                        st.metric(
                            "Abnormal Probability",
                            f"{result['probability_abnormal']*100:.2f}%"
                        )
                    
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
                            "- Check features with high deviation scores\n"
                            "- Compare with similar abnormal examples\n"
                            "- Review maintenance records\n"
                            "- Consider backup pump activation"
                        )
                    
                    # Raw results (expandable)
                    with st.expander("🔍 View Raw Results"):
                        st.json(result)
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
    
    else:
        # Instructions when no file uploaded
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
                "Advanced ML model with explainability analyzes patterns"
            )
        
        with col3:
            st.markdown(
                "#### 3️⃣ Get Insights\n"
                "Receive diagnosis with similarity-based explanations"
            )
        
        st.markdown("---")
        st.markdown("### 🧠 AI Explainability Features")
        st.info(
            "**This system shows you WHY the AI made its decision:**\n\n"
            "- **Feature Space Visualization**: See where your pump sits compared to training data\n"
            "- **Similar Examples**: View the most similar pump sounds from training\n"
            "- **Distance Metrics**: Quantify how close you are to normal vs abnormal clusters\n"
            "- **Feature Importance**: Understand which acoustic features drove the decision\n\n"
            "**Transparent, verifiable, and trustworthy AI predictions!**"
        )

if __name__ == "__main__":
    main()