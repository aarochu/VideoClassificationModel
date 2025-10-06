# SmartFace - Real-time Face Detection & Tracking MVP

A real-time face detection and tracking web application that detects all faces in video frames, draws green bounding boxes around each face, and assigns unique labels that stay consistent as people move.

## 🚀 MVP Features

### Core Features ✅
- **Face Detection**: Detect faces in each frame using ML (MediaPipe/OpenCV)
- **Face Tracking**: Track same face across frames with consistent labels
- **Bounding Boxes**: Green boxes drawn in real time around detected faces
- **Label Assignment**: Assign and persist "Person 1, 2, 3..." labels
- **Web UI**: Upload videos or use webcam with real-time visualization

### Stretch Goals 🚀
- **Identity Recognition**: Match faces to known people
- **Emotion Detection**: Detect emotions (happy, sad, etc.)
- **Analytics Dashboard**: Track engagement and attendance
- **Deployment**: Deploy on cloud platforms

## 🎯 Project Overview

This project provides both:
1. **SmartFace MVP**: Real-time face detection and tracking system
2. **Human Classifier**: Deep learning model for identity/activity classification

## 🏗️ Architecture

- **Frontend**: React web app for video upload and prediction display
- **Backend**: FastAPI server for model inference
- **Model**: CNN+LSTM architecture for video classification
- **Data**: Video clips organized by class in folder structure

## 📁 Project Structure

```
VideoClassificationModel/
├── dataset/                 # Training data
│   ├── PersonA/            # Identity classification
│   ├── PersonB/
│   └── PersonC/
├── src/
│   ├── model/              # Model architecture and training
│   ├── data/               # Data loading and preprocessing
│   ├── api/                # FastAPI backend
│   └── frontend/           # React frontend
├── models/                 # Saved model weights
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Option 1: SmartFace Demo (Fastest)
```bash
# Install dependencies
pip install -r requirements.txt

# Run standalone face detection demo
python3 smartface_demo.py

# Or with specific options
python3 smartface_demo.py --method mediapipe --camera 0
```

### Option 2: Full Web Application
```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend API
python3 src/api/main.py

# In another terminal, start the frontend
cd src/frontend
npm install
npm start
```

### Option 3: Human Classifier Training
```bash
# Run the setup script
python3 setup.py

# Create sample dataset structure
python3 create_sample_data.py

# Add your video data to dataset/PersonA/, dataset/PersonB/, etc.

# Train the model
python3 src/model/train.py

# Run the complete demo
python3 run_demo.py
```

### Option 2: Manual Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install frontend dependencies**:
   ```bash
   cd src/frontend
   npm install
   ```

3. **Prepare your data**:
   ```bash
   python3 create_sample_data.py
   # Add 10-20 short video clips per class to dataset/ directories
   ```

4. **Train the model**:
   ```bash
   python3 src/model/train.py
   ```

5. **Start the backend**:
   ```bash
   python3 src/api/main.py
   ```

6. **Start the frontend** (in a new terminal):
   ```bash
   cd src/frontend
   npm start
   ```

### Option 3: Quick Demo
```bash
# If you already have a trained model
python3 run_demo.py
```

## 🧠 Model Architecture

- **CNN Backbone**: ResNet18 for feature extraction from video frames
- **LSTM**: Processes temporal sequence of CNN features
- **Output**: Softmax probabilities for each class

## 📊 Data Format

- Input: 5-10 second video clips
- Frame size: 224×224×3
- Frame rate: 30 FPS (configurable)
- Classes: Organized in folders by label

## 🔧 API Endpoints

### SmartFace Detection
- `POST /detect_faces`: Detect faces in uploaded image
- `POST /detect_faces_video`: Detect faces in uploaded video
- `GET /face_detector_status`: Get face detector status

### Human Classifier
- `POST /predict`: Upload video and get classification
- `GET /health`: Health check
- `GET /classes`: List available classes

## 🎨 Frontend Features

### SmartFace Detection
- Real-time face detection with webcam
- Green bounding boxes around detected faces
- Consistent person labeling across frames
- Live statistics and detection counts

### Human Classifier
- Video upload interface
- Real-time prediction display
- Confidence scores
- Video preview

## ⚙️ Tech Stack

### Backend (AI Model)
- **MediaPipe**: High-accuracy face detection (primary)
- **OpenCV**: Fast face detection with Haar Cascades (fallback)
- **FastAPI**: REST API for face detection endpoints
- **Python**: Core processing and ML pipeline

### Frontend
- **React**: Modern web interface
- **Styled Components**: Component styling
- **Canvas API**: Real-time face box rendering
- **WebRTC**: Camera access and video streaming

### Face Detection Methods
1. **MediaPipe Face Detection** (Recommended)
   - Very accurate face detection
   - Good performance on modern hardware
   - Handles various lighting conditions

2. **OpenCV Haar Cascades** (Fallback)
   - Fast and lightweight
   - Works on older hardware
   - Good for basic face detection

## 💡 Usage Examples

### Standalone Demo
```bash
# Basic face detection
python3 smartface_demo.py

# Use OpenCV instead of MediaPipe
python3 smartface_demo.py --method opencv

# Record detection session
python3 smartface_demo.py --output detection_session.mp4
```

### Web Application
1. Start backend: `python3 src/api/main.py`
2. Start frontend: `cd src/frontend && npm start`
3. Open browser to `http://localhost:3000`
4. Click "🚀 SmartFace Detection" tab
5. Allow camera permissions and start detection

### API Usage
```python
import requests

# Detect faces in image
with open('photo.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/detect_faces', 
                           files={'file': f})
    result = response.json()
    print(f"Found {result['faces_detected']} faces")
```

## 🎯 Why It's Valuable

- **🔒 Security**: Detect unauthorized people in real-time
- **📊 Analytics**: Track engagement or attendance automatically  
- **👋 Personalization**: Recognize returning users for custom experiences
- **💡 Scalable**: Foundation for human recognition or safety AI applications
