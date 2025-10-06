"""
Face Detection and Tracking Module

This module provides real-time face detection and tracking capabilities using OpenCV
and MediaPipe. It maintains consistent face labels across video frames and provides
bounding box coordinates for each detected face.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict
import math

# Try to import MediaPipe, fall back to OpenCV if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available, using OpenCV only")


class FaceTracker:
    """
    Face tracking system that maintains consistent labels for faces across frames.
    Uses IoU (Intersection over Union) to match faces between frames.
    """
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 0.5):
        """
        Initialize face tracker.
        
        Args:
            max_disappeared: Maximum frames a face can be missing before being removed
            max_distance: Maximum IoU distance for face matching
        """
        self.next_face_id = 1
        self.faces = {}  # face_id -> face_data
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_distance(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between centers of two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        max_dimension = max(w1, h1, w2, h2)
        
        return distance / max_dimension if max_dimension > 0 else float('inf')
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """
        Update face tracker with new detections.
        
        Args:
            detections: List of (x, y, width, height) bounding boxes
            
        Returns:
            List of face data with consistent IDs
        """
        current_time = time.time()
        
        # If no faces detected, mark all as disappeared
        if len(detections) == 0:
            for face_id in list(self.faces.keys()):
                self.faces[face_id]['disappeared'] += 1
                if self.faces[face_id]['disappeared'] > self.max_disappeared:
                    del self.faces[face_id]
            return []
        
        # If no existing faces, create new ones
        if len(self.faces) == 0:
            for detection in detections:
                face_id = self.next_face_id
                self.next_face_id += 1
                self.faces[face_id] = {
                    'bbox': detection,
                    'disappeared': 0,
                    'first_seen': current_time,
                    'last_seen': current_time
                }
        else:
            # Match detections to existing faces
            face_ids = list(self.faces.keys())
            detection_indices = list(range(len(detections)))
            
            # Calculate distance matrix
            distances = []
            for face_id in face_ids:
                face_bbox = self.faces[face_id]['bbox']
                face_distances = []
                for detection in detections:
                    # Use IoU as similarity measure (higher is better)
                    iou = self.calculate_iou(face_bbox, detection)
                    distance = 1.0 - iou  # Convert to distance (lower is better)
                    face_distances.append(distance)
                distances.append(face_distances)
            
            # Simple greedy matching
            matched_faces = set()
            matched_detections = set()
            
            # Sort by distance and match
            for face_idx, face_id in enumerate(face_ids):
                if face_id in matched_faces:
                    continue
                    
                best_detection = None
                best_distance = float('inf')
                
                for det_idx in detection_indices:
                    if det_idx in matched_detections:
                        continue
                        
                    distance = distances[face_idx][det_idx]
                    if distance < best_distance and distance < self.max_distance:
                        best_distance = distance
                        best_detection = det_idx
                
                if best_detection is not None:
                    # Update existing face
                    self.faces[face_id]['bbox'] = detections[best_detection]
                    self.faces[face_id]['disappeared'] = 0
                    self.faces[face_id]['last_seen'] = current_time
                    matched_faces.add(face_id)
                    matched_detections.add(best_detection)
                else:
                    # Mark as disappeared
                    self.faces[face_id]['disappeared'] += 1
            
            # Create new faces for unmatched detections
            for det_idx, detection in enumerate(detections):
                if det_idx not in matched_detections:
                    face_id = self.next_face_id
                    self.next_face_id += 1
                    self.faces[face_id] = {
                        'bbox': detection,
                        'disappeared': 0,
                        'first_seen': current_time,
                        'last_seen': current_time
                    }
            
            # Remove faces that have been missing too long
            for face_id in list(self.faces.keys()):
                if self.faces[face_id]['disappeared'] > self.max_disappeared:
                    del self.faces[face_id]
        
        # Return current face data
        result = []
        for face_id, face_data in self.faces.items():
            if face_data['disappeared'] == 0:  # Only return currently visible faces
                result.append({
                    'id': face_id,
                    'label': f'Person {face_id}',
                    'bbox': face_data['bbox'],
                    'confidence': 0.9,  # MediaPipe provides high confidence
                    'first_seen': face_data['first_seen'],
                    'last_seen': face_data['last_seen']
                })
        
        return result


class FaceDetector:
    """
    Face detection system using MediaPipe for high accuracy and speed.
    Falls back to OpenCV if MediaPipe is not available.
    """
    
    def __init__(self, model_selection: int = 0, min_detection_confidence: float = 0.5):
        """
        Initialize face detector.
        
        Args:
            model_selection: 0 for short-range (2m), 1 for full-range (5m)
            min_detection_confidence: Minimum confidence threshold
        """
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
            self.use_mediapipe = True
        else:
            # Fall back to OpenCV
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_mediapipe = False
        
        self.tracker = FaceTracker()
        
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a frame and return with tracking information.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of face detection results with tracking IDs
        """
        detections = []
        
        if self.use_mediapipe:
            # Use MediaPipe for detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                h, w, _ = frame.shape
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert to absolute coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    detections.append((x, y, width, height))
        else:
            # Use OpenCV for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                detections.append((x, y, w, h))
        
        # Update tracker and return results
        tracked_faces = self.tracker.update(detections)
        return tracked_faces
    
    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input frame
            faces: List of face detection results
            
        Returns:
            Frame with drawn bounding boxes and labels
        """
        result_frame = frame.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            label = face['label']
            confidence = face['confidence']
            
            # Draw bounding box (green)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label background
            label_text = f"{label} ({confidence:.1%})"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background rectangle
            cv2.rectangle(
                result_frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_frame,
                label_text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
        
        return result_frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame: detect faces, track them, and draw results.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, face_detections)
        """
        faces = self.detect_faces(frame)
        annotated_frame = self.draw_faces(frame, faces)
        return annotated_frame, faces


class OpenCVFaceDetector:
    """
    Alternative face detector using OpenCV Haar Cascades.
    Faster but less accurate than MediaPipe.
    """
    
    def __init__(self):
        """Initialize OpenCV face detector."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.tracker = FaceTracker()
        
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using OpenCV Haar Cascades.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of face detection results with tracking IDs
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append((x, y, w, h))
        
        # Update tracker and return results
        tracked_faces = self.tracker.update(detections)
        return tracked_faces
    
    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input frame
            faces: List of face detection results
            
        Returns:
            Frame with drawn bounding boxes and labels
        """
        result_frame = frame.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            label = face['label']
            confidence = face['confidence']
            
            # Draw bounding box (green)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label background
            label_text = f"{label} ({confidence:.1%})"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background rectangle
            cv2.rectangle(
                result_frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_frame,
                label_text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
        
        return result_frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame: detect faces, track them, and draw results.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, face_detections)
        """
        faces = self.detect_faces(frame)
        annotated_frame = self.draw_faces(frame, faces)
        return annotated_frame, faces


def create_face_detector(method: str = "mediapipe") -> FaceDetector:
    """
    Factory function to create face detector.
    
    Args:
        method: "mediapipe" or "opencv"
        
    Returns:
        Face detector instance
    """
    if method.lower() == "mediapipe":
        if MEDIAPIPE_AVAILABLE:
            return FaceDetector()
        else:
            print("MediaPipe not available, falling back to OpenCV")
            return OpenCVFaceDetector()
    elif method.lower() == "opencv":
        return OpenCVFaceDetector()
    else:
        raise ValueError(f"Unknown face detection method: {method}")


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    # Create face detector
    detector = create_face_detector("mediapipe")
    
    print("Face detection started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, faces = detector.process_frame(frame)
        
        # Display results
        cv2.imshow('SmartFace - Face Detection & Tracking', annotated_frame)
        
        # Print face count
        if faces:
            print(f"Detected {len(faces)} faces: {[f['label'] for f in faces]}")
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
