"""
# Pump-Net: Industrial Pump Anomaly Detection System

## 🎯 Overview
Pump-Net is a production-ready machine learning system for detecting anomalies in industrial pump sounds. It achieves 98.07% accuracy using engineered acoustic features and a deep neural network.

## 🏗️ Architecture
- **Backend**: FastAPI RESTful API
- **Frontend**: Streamlit interactive dashboard
- **ML Model**: Feature-based Deep Neural Network (DNN)
- **Features**: MFCCs, Spectral, and Temporal features

## 📁 Project Structure
```
pump-net-production/
├── config/          # Configuration settings
├── data/            # Data and models
├── src/             # Core ML pipeline
├── api/             # FastAPI backend
├── ui/              # Streamlit frontend
├── utils/           # Utilities
└── tests/           # Unit tests
```

## 🚀 Quick Start

### 1. Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training
```bash
# Train the model
python -m src.train
```

### 3. Start API Server
```bash
# Terminal 1: Start FastAPI backend
python api/main.py

# Or using uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start UI
```bash
# Terminal 2: Start Streamlit frontend
streamlit run ui/streamlit_app.py --server.port 8501
```

### 5. Access Application
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **UI**: http://localhost:8501

## 🎯 Usage

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \\
  -F "file=@path/to/audio.wav"
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch" \\
  -F "files=@audio1.wav" \\
  -F "files=@audio2.wav"
```

### Python API
```python
from src.inference import PumpNetInference

# Initialize
inference = PumpNetInference()

# Predict
result = inference.predict_from_file("audio.wav")
print(result)
```

## 📊 Performance Metrics
- **Accuracy**: 98.07%
- **Precision**: 100.00%
- **Recall**: 91.23%
- **F1-Score**: 95.41%
- **AUC-ROC**: 95.77%

## 🔧 Configuration
Edit `config/config.py` or create `.env` file:
```bash
API_HOST=0.0.0.0
API_PORT=8000
PREDICTION_THRESHOLD=0.5
CONFIDENCE_THRESHOLD=0.7
```

## 🧪 Testing
```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 📦 Deployment

### Docker
```bash
# Build image
docker build -t pump-net:latest .

# Run container
docker run -p 8000:8000 -p 8501:8501 pump-net:latest
```

### Production Considerations
- Use Gunicorn/Uvicorn workers for API
- Enable HTTPS with reverse proxy (Nginx)
- Implement rate limiting
- Add authentication/authorization
- Monitor with Prometheus + Grafana
- Set up logging aggregation

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License
MIT License

## 👥 Authors
Pump-Net Development Team

## 🙏 Acknowledgments
- MIMII Dataset for training data
- Librosa for audio processing
- TensorFlow/Keras for deep learning
"""
