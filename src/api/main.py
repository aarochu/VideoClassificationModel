import os
import sys
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from pathlib import Path
import tempfile
from typing import List, Dict, Any
import json
from datetime import datetime
import shutil
import io
import base64

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.video_classifier import VideoClassifier, VideoClassifierConfig
from data.dataset import VideoDataset
from model.face_detector import create_face_detector, FaceDetector, OpenCVFaceDetector


class VideoClassifierAPI:
    """API wrapper for VideoClassifier model."""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config and classes
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            self.config = VideoClassifierConfig(**config_dict)
        else:
            # Default config if not found
            self.config = VideoClassifierConfig()
        
        self.classes = checkpoint.get('classes', ['Person1', 'Person2', 'Person3'])
        self.config.num_classes = len(self.classes)
        
        # Create and load model
        self.model = VideoClassifier(
            num_classes=self.config.num_classes,
            sequence_length=self.config.sequence_length,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            pretrained=False  # Don't load pretrained weights for inference
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.classes}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        Preprocess video file for inference.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed video tensor
        """
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        if total_frames <= self.config.sequence_length:
            # If video is shorter than required sequence, repeat frames
            frame_indices = list(range(total_frames))
            while len(frame_indices) < self.config.sequence_length:
                frame_indices.extend(list(range(total_frames)))
            frame_indices = frame_indices[:self.config.sequence_length]
        else:
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.config.sequence_length, dtype=int)
        
        # Extract and preprocess frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(frame, self.config.frame_size)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                # Normalize with ImageNet stats
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                frame = (frame - mean) / std
                frames.append(frame)
            else:
                # If frame extraction fails, use the last successful frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # If no frames extracted, create a black frame
                    frames.append(np.zeros((*self.config.frame_size, 3), dtype=np.float32))
        
        cap.release()
        
        # Convert to tensor and add batch dimension
        video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)  # (T, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        
        return video_tensor.to(self.device)
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Make prediction on video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess video
            video_tensor = self.preprocess_video(video_path)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(video_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()[0]
            predicted_class = predicted_class.cpu().item()
            confidence = confidence.cpu().item()
            
            # Create results
            results = {
                "predicted_class": self.classes[predicted_class],
                "confidence": float(confidence),
                "all_probabilities": {
                    self.classes[i]: float(prob) for i, prob in enumerate(probabilities)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="Human Classifier API",
    description="API for classifying people in video clips",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_api = None
face_detector = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model_api, face_detector
    
    # Initialize face detector
    try:
        face_detector = create_face_detector("mediapipe")
        print("Face detector initialized successfully")
    except Exception as e:
        print(f"Failed to initialize face detector: {e}")
        try:
            face_detector = create_face_detector("opencv")
            print("OpenCV face detector initialized as fallback")
        except Exception as e2:
            print(f"Failed to initialize OpenCV face detector: {e2}")
            face_detector = None
    
    # Look for the best model
    models_dir = Path("../../models")
    best_model_path = None
    
    if models_dir.exists():
        # Look for best_model.pth
        best_model_path = models_dir / "best_model.pth"
        if not best_model_path.exists():
            # Look for any checkpoint
            checkpoints = list(models_dir.glob("*.pth"))
            if checkpoints:
                best_model_path = checkpoints[0]
    
    if best_model_path and best_model_path.exists():
        try:
            model_api = VideoClassifierAPI(str(best_model_path))
            print(f"Model loaded from: {best_model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            model_api = None
    else:
        print("No trained model found. Please train a model first.")
        print("Run: python src/model/train.py")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Human Classifier API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_api is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_api is not None,
        "device": str(model_api.device) if model_api else None
    }


@app.get("/classes")
async def get_classes():
    """Get list of available classes."""
    if model_api is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": model_api.classes,
        "num_classes": len(model_api.classes)
    }


@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """
    Predict the class of a video file.
    
    Args:
        file: Video file to classify
        
    Returns:
        Prediction results
    """
    if model_api is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Make prediction
        results = model_api.predict(tmp_file_path)
        
        return JSONResponse(content=results)
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict the class of multiple video files.
    
    Args:
        files: List of video files to classify
        
    Returns:
        List of prediction results
    """
    if model_api is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            results.append({
                "filename": file.filename,
                "error": "File must be a video"
            })
            continue
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Make prediction
            prediction = model_api.predict(tmp_file_path)
            prediction["filename"] = file.filename
            results.append(prediction)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    return JSONResponse(content={"results": results})


@app.post("/save_video")
async def save_video(
    file: UploadFile = File(...),
    class_name: str = Form(...)
):
    """
    Save a recorded video to the dataset.
    
    Args:
        file: Video file to save
        class_name: Class name to save the video under
        
    Returns:
        Success message
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create dataset directory structure
    dataset_dir = Path("../../dataset")
    class_dir = dataset_dir / class_name
    
    # Create class directory if it doesn't exist
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{class_name.lower()}_{timestamp}.webm"
    file_path = class_dir / filename
    
    try:
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return JSONResponse(content={
            "message": f"Video saved successfully to {class_name} class",
            "filename": filename,
            "path": str(file_path),
            "class": class_name
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")


@app.get("/dataset_info")
async def get_dataset_info():
    """
    Get information about the current dataset.
    
    Returns:
        Dataset information including classes and video counts
    """
    dataset_dir = Path("../../dataset")
    
    if not dataset_dir.exists():
        return JSONResponse(content={
            "exists": False,
            "message": "Dataset directory does not exist"
        })
    
    classes_info = {}
    total_videos = 0
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            class_name = class_dir.name
            video_files = []
            
            # Count video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            for ext in video_extensions:
                video_files.extend(list(class_dir.glob(f"*{ext}")))
                video_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
            
            classes_info[class_name] = {
                "video_count": len(video_files),
                "videos": [f.name for f in video_files]
            }
            total_videos += len(video_files)
    
    return JSONResponse(content={
        "exists": True,
        "total_videos": total_videos,
        "classes": classes_info,
        "dataset_path": str(dataset_dir)
    })


@app.post("/detect_faces")
async def detect_faces_in_image(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image.
    
    Args:
        file: Image file to analyze
        
    Returns:
        Face detection results with bounding boxes and labels
    """
    if face_detector is None:
        raise HTTPException(status_code=503, detail="Face detector not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        # Draw faces on image
        annotated_image = face_detector.draw_faces(image, faces)
        
        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "faces_detected": len(faces),
            "faces": [
                {
                    "id": face["id"],
                    "label": face["label"],
                    "bbox": {
                        "x": int(face["bbox"][0]),
                        "y": int(face["bbox"][1]),
                        "width": int(face["bbox"][2]),
                        "height": int(face["bbox"][3])
                    },
                    "confidence": float(face["confidence"])
                }
                for face in faces
            ],
            "annotated_image": f"data:image/jpeg;base64,{image_base64}",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")


@app.post("/detect_faces_video")
async def detect_faces_in_video(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded video file.
    
    Args:
        file: Video file to analyze
        
    Returns:
        Face detection results for each frame
    """
    if face_detector is None:
        raise HTTPException(status_code=503, detail="Face detector not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Open video
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        frame_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces in frame
            faces = face_detector.detect_faces(frame)
            
            frame_results.append({
                "frame": frame_count,
                "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                "faces": [
                    {
                        "id": face["id"],
                        "label": face["label"],
                        "bbox": {
                            "x": int(face["bbox"][0]),
                            "y": int(face["bbox"][1]),
                            "width": int(face["bbox"][2]),
                            "height": int(face["bbox"][3])
                        },
                        "confidence": float(face["confidence"])
                    }
                    for face in faces
                ]
            })
            
            frame_count += 1
            
            # Limit processing to avoid timeout
            if frame_count > 1000:  # Process max 1000 frames
                break
        
        cap.release()
        
        return JSONResponse(content={
            "total_frames": frame_count,
            "fps": cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0,
            "frames": frame_results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video face detection failed: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.get("/face_detector_status")
async def get_face_detector_status():
    """Get face detector status and capabilities."""
    return JSONResponse(content={
        "available": face_detector is not None,
        "type": type(face_detector).__name__ if face_detector else None,
        "capabilities": {
            "image_detection": face_detector is not None,
            "video_detection": face_detector is not None,
            "face_tracking": face_detector is not None,
            "real_time": face_detector is not None
        }
    })


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
