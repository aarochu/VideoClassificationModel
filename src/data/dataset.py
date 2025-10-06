import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict, Optional
import random
from pathlib import Path


class VideoDataset(Dataset):
    """
    Dataset class for video classification.
    
    Loads video files and extracts frames for training.
    """
    
    def __init__(
        self,
        data_dir: str,
        classes: List[str],
        sequence_length: int = 30,
        frame_size: Tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None,
        mode: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.mode = mode
        
        # Default transforms
        if transform is None:
            if mode == "train":
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(frame_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(frame_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Load video file paths and labels
        self.video_paths = []
        self.labels = []
        
        self._load_video_paths()
        
    def _load_video_paths(self):
        """Load all video file paths and their corresponding labels."""
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} does not exist")
                continue
                
            # Get all video files in the class directory
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(list(class_dir.glob(f"*{ext}")))
                video_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
            
            # Add to dataset
            for video_file in video_files:
                self.video_paths.append(str(video_file))
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.video_paths)} videos from {len(self.classes)} classes")
        for i, class_name in enumerate(self.classes):
            count = sum(1 for label in self.labels if label == i)
            print(f"  {class_name}: {count} videos")
    
    def _extract_frames(self, video_path: str) -> np.ndarray:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of frames with shape (num_frames, height, width, channels)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to sample
        if total_frames <= self.sequence_length:
            # If video is shorter than required sequence, repeat frames
            frame_indices = list(range(total_frames))
            while len(frame_indices) < self.sequence_length:
                frame_indices.extend(list(range(total_frames)))
            frame_indices = frame_indices[:self.sequence_length]
        else:
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame extraction fails, use the last successful frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # If no frames extracted, create a black frame
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        cap.release()
        
        return np.array(frames)
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a video sequence and its label.
        
        Args:
            idx: Index of the video
            
        Returns:
            Tuple of (video_tensor, label)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract frames
            frames = self._extract_frames(video_path)
            
            # Apply transforms
            transformed_frames = []
            for frame in frames:
                if self.transform:
                    frame_tensor = self.transform(frame)
                else:
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                transformed_frames.append(frame_tensor)
            
            # Stack frames into sequence tensor
            video_tensor = torch.stack(transformed_frames)
            
            return video_tensor, label
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return a zero tensor as fallback
            video_tensor = torch.zeros(self.sequence_length, 3, *self.frame_size)
            return video_tensor, label


def create_data_loaders(
    data_dir: str,
    classes: List[str],
    batch_size: int = 8,
    sequence_length: int = 30,
    frame_size: Tuple[int, int] = (224, 224),
    train_split: float = 0.8,
    num_workers: int = 4,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Directory containing video data
        classes: List of class names
        batch_size: Batch size for data loaders
        sequence_length: Number of frames per video sequence
        frame_size: Size to resize frames to
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = VideoDataset(
        data_dir=data_dir,
        classes=classes,
        sequence_length=sequence_length,
        frame_size=frame_size,
        mode="train"
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_classes_from_directory(data_dir: str) -> List[str]:
    """
    Get list of classes from directory structure.
    
    Args:
        data_dir: Directory containing class subdirectories
        
    Returns:
        List of class names
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    classes = []
    for item in data_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            classes.append(item.name)
    
    classes.sort()  # Sort for consistent ordering
    return classes


def create_sample_dataset(data_dir: str, num_classes: int = 3, videos_per_class: int = 5):
    """
    Create a sample dataset structure for testing.
    
    Args:
        data_dir: Directory to create sample data in
        num_classes: Number of classes to create
        videos_per_class: Number of sample videos per class
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    for i in range(num_classes):
        class_dir = data_path / f"Person{i+1}"
        class_dir.mkdir(exist_ok=True)
        
        print(f"Created class directory: {class_dir}")
        print(f"  Add {videos_per_class} video files to this directory")


if __name__ == "__main__":
    # Test the dataset
    data_dir = "../../dataset"
    
    try:
        classes = get_classes_from_directory(data_dir)
        print(f"Found classes: {classes}")
        
        if classes:
            dataset = VideoDataset(data_dir, classes, sequence_length=10)
            print(f"Dataset size: {len(dataset)}")
            
            if len(dataset) > 0:
                video_tensor, label = dataset[0]
                print(f"Video tensor shape: {video_tensor.shape}")
                print(f"Label: {label}")
                print("Dataset test passed!")
            else:
                print("No videos found in dataset")
        else:
            print("No classes found. Creating sample structure...")
            create_sample_dataset(data_dir)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Creating sample dataset structure...")
        create_sample_dataset(data_dir)
