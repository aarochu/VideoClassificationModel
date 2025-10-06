#!/usr/bin/env python3
"""
SmartFace Demo - Standalone Face Detection and Tracking

This script demonstrates the face detection and tracking capabilities
using OpenCV and MediaPipe. It can run independently to test the
face detection system.

Usage:
    python smartface_demo.py [--method mediapipe|opencv] [--camera 0]
"""

import cv2
import argparse
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model.face_detector import create_face_detector


def main():
    parser = argparse.ArgumentParser(description='SmartFace Demo - Face Detection & Tracking')
    parser.add_argument('--method', choices=['mediapipe', 'opencv'], default='mediapipe',
                       help='Face detection method to use (default: mediapipe)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index to use (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path (optional)')
    
    args = parser.parse_args()
    
    print("🚀 SmartFace Demo - Real-time Face Detection & Tracking")
    print(f"Method: {args.method}")
    print(f"Camera: {args.camera}")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Create face detector
    try:
        detector = create_face_detector(args.method)
        print(f"✅ {args.method.title()} face detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize {args.method} detector: {e}")
        if args.method == 'mediapipe':
            print("🔄 Falling back to OpenCV detector...")
            try:
                detector = create_face_detector('opencv')
                print("✅ OpenCV face detector initialized successfully")
            except Exception as e2:
                print(f"❌ Failed to initialize OpenCV detector: {e2}")
                return 1
        else:
            return 1
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {args.camera}")
        return 1
    
    # Set camera properties for better face detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Camera: {width}x{height} @ {fps:.1f} FPS")
    
    # Video writer for output
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"💾 Recording to: {args.output}")
    
    frame_count = 0
    total_faces_detected = 0
    unique_persons = set()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Process frame for face detection
            annotated_frame, faces = detector.process_frame(frame)
            
            # Update statistics
            if faces:
                total_faces_detected += len(faces)
                for face in faces:
                    unique_persons.add(face['id'])
            
            # Add statistics overlay
            stats_text = [
                f"Frame: {frame_count}",
                f"Faces: {len(faces)}",
                f"Total Detections: {total_faces_detected}",
                f"Unique Persons: {len(unique_persons)}",
                f"Method: {args.method.title()}"
            ]
            
            y_offset = 30
            for text in stats_text:
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y_offset += 25
            
            # Add instructions
            instructions = [
                "Press 'q' to quit",
                "Press 's' to save screenshot"
            ]
            
            y_offset = height - 60
            for instruction in instructions:
                cv2.putText(annotated_frame, instruction, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, instruction, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 20
            
            # Write to output video if specified
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display the frame
            cv2.imshow('SmartFace - Face Detection & Tracking', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('s'):
                screenshot_path = f"smartface_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            
            # Print face detection info every 30 frames
            if frame_count % 30 == 0 and faces:
                face_labels = [face['label'] for face in faces]
                print(f"Frame {frame_count}: Detected {len(faces)} faces - {', '.join(face_labels)}")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n📊 Final Statistics:")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Total face detections: {total_faces_detected}")
        print(f"   Unique persons detected: {len(unique_persons)}")
        if frame_count > 0:
            print(f"   Average faces per frame: {total_faces_detected / frame_count:.2f}")
        print(f"   Detection method: {args.method.title()}")
        
        if args.output and video_writer:
            print(f"   Video saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
