import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class VideoClassifier(nn.Module):
    """
    CNN+LSTM model for video classification.
    
    Architecture:
    1. ResNet18 backbone for feature extraction from frames
    2. LSTM for temporal modeling
    3. Classification head for final prediction
    """
    
    def __init__(
        self,
        num_classes: int,
        sequence_length: int = 30,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True
    ):
        super(VideoClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # CNN backbone (ResNet18)
        self.cnn_backbone = models.resnet18(pretrained=pretrained)
        # Remove the final classification layer
        self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-1])
        
        # Get the feature dimension from ResNet18
        self.feature_dim = 512
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        lstm_output_size = hidden_size * 2  # *2 for bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features using CNN backbone
        with torch.no_grad():
            features = self.cnn_backbone(x)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)
        
        # Process sequence with LSTM
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Use the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(last_output)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from video frames.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Features tensor of shape (batch_size, sequence_length, feature_dim)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features using CNN backbone
        with torch.no_grad():
            features = self.cnn_backbone(x)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)
        
        return features


class VideoClassifierConfig:
    """Configuration class for VideoClassifier model."""
    
    def __init__(
        self,
        num_classes: int = 3,
        sequence_length: int = 30,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        num_epochs: int = 50,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.frame_size = frame_size


def create_model(config: VideoClassifierConfig) -> VideoClassifier:
    """
    Create a VideoClassifier model with the given configuration.
    
    Args:
        config: VideoClassifierConfig object
        
    Returns:
        Initialized VideoClassifier model
    """
    model = VideoClassifier(
        num_classes=config.num_classes,
        sequence_length=config.sequence_length,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    config = VideoClassifierConfig(num_classes=3)
    model = create_model(config)
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    batch_size = 2
    sequence_length = config.sequence_length
    channels, height, width = 3, config.frame_size[0], config.frame_size[1]
    
    x = torch.randn(batch_size, sequence_length, channels, height, width)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
