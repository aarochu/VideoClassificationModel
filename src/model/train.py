import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from typing import Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.video_classifier import VideoClassifier, VideoClassifierConfig, create_model
from data.dataset import create_data_loaders, get_classes_from_directory


class Trainer:
    """Training class for VideoClassifier model."""
    
    def __init__(self, config: VideoClassifierConfig, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load classes
        self.classes = get_classes_from_directory(data_dir)
        self.config.num_classes = len(self.classes)
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        # Create model
        self.model = create_model(self.config).to(self.device)
        print(f"Model created with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} parameters")
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            data_dir=data_dir,
            classes=self.classes,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            frame_size=config.frame_size,
            num_workers=2  # Reduce for stability
        )
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Setup logging
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'classes': self.classes
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Save final model
        self.save_checkpoint(self.config.num_epochs - 1, False)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config.__dict__,
            'classes': self.classes
        }
        
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model and logs saved to: {self.output_dir}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Video Classifier")
    parser.add_argument("--data_dir", type=str, default="../../dataset", 
                       help="Directory containing video data")
    parser.add_argument("--output_dir", type=str, default="../../models", 
                       help="Directory to save model and logs")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, 
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--sequence_length", type=int, default=30, 
                       help="Number of frames per video sequence")
    parser.add_argument("--hidden_size", type=int, default=512, 
                       help="LSTM hidden size")
    
    args = parser.parse_args()
    
    # Create config
    config = VideoClassifierConfig(
        num_classes=0,  # Will be set based on data
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"training_{timestamp}"
    
    # Create trainer and start training
    trainer = Trainer(config, args.data_dir, str(output_dir))
    trainer.train()


if __name__ == "__main__":
    main()
