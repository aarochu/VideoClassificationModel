# Human Classifier Dataset

This directory contains the training data for the Human Classifier model.

## Directory Structure:
```
dataset/
├── PersonA/          # Videos of Person A
├── PersonB/          # Videos of Person B
└── PersonC/          # Videos of Person C
```

## Data Requirements:
- **Format**: MP4, AVI, MOV, MKV, WMV, FLV
- **Duration**: 5-10 seconds per video
- **Quantity**: 10-20 videos per class
- **Quality**: Clear, well-lit videos with consistent subjects

## Adding Your Data:
1. Place video files in the appropriate class directory
2. Ensure consistent naming (e.g., personA_video1.mp4)
3. Remove any README.md files from class directories after adding videos
4. Run the training script: `python src/model/train.py`

## Alternative: Behavior Classification
Instead of person identity, you can classify behaviors:
- `Sitting/`
- `Walking/`
- `Dancing/`
- `Running/`

Just replace the PersonA/PersonB/PersonC directories with your behavior classes.
