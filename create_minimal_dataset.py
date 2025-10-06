#!/usr/bin/env python3
"""
Create a minimal dataset for testing the video classification system.
This script creates sample video files for training.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import tempfile

def create_sample_video(output_path: str, duration: int = 3, fps: int = 30, width: int = 640, height: int = 480):
    """
    Create a sample video file with a simple pattern.
    
    Args:
        output_path: Path where to save the video
        duration: Duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create a simple pattern that changes over time
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a moving circle
        center_x = int(width * 0.5 + 100 * np.sin(frame_num * 0.1))
        center_y = int(height * 0.5 + 50 * np.cos(frame_num * 0.1))
        radius = 50
        
        # Draw a circle with changing color
        color = (
            int(255 * (0.5 + 0.5 * np.sin(frame_num * 0.05))),
            int(255 * (0.5 + 0.5 * np.cos(frame_num * 0.07))),
            int(255 * (0.5 + 0.5 * np.sin(frame_num * 0.03)))
        )
        
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
        
        # Add some text
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created sample video: {output_path}")

def create_minimal_dataset():
    """Create a minimal dataset for testing."""
    dataset_dir = Path("dataset")
    
    # Create class directories
    classes = ["PersonA", "PersonB", "PersonC"]
    
    for class_name in classes:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Remove README files
        readme_file = class_dir / "README.md"
        if readme_file.exists():
            readme_file.unlink()
        
        # Create 3 sample videos for each class
        for i in range(3):
            video_path = class_dir / f"{class_name.lower()}_sample_{i+1}.mp4"
            create_sample_video(str(video_path), duration=2, fps=15)
    
    print("Minimal dataset created successfully!")
    print("Each class now has 3 sample videos for training.")

if __name__ == "__main__":
    create_minimal_dataset()
