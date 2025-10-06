#!/usr/bin/env python3
"""
Script to create sample dataset structure for the Human Classifier project.
This script creates the directory structure and provides instructions for adding video data.
"""

import os
from pathlib import Path


def create_sample_dataset():
    """Create sample dataset structure."""
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Create sample classes
    classes = ["PersonA", "PersonB", "PersonC"]
    
    for class_name in classes:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create a README file in each class directory
        readme_content = f"""# {class_name} Videos

This directory should contain video files for {class_name}.

## Requirements:
- Video format: MP4, AVI, MOV, MKV, WMV, or FLV
- Duration: 5-10 seconds recommended
- Resolution: Any (will be resized to 224x224)
- Content: Clear view of the person/activity

## Recommended:
- 10-20 videos per class for good training results
- Consistent lighting and background
- Different angles and poses for better generalization

## Example files:
- {class_name.lower()}_video1.mp4
- {class_name.lower()}_video2.mp4
- {class_name.lower()}_video3.mp4
"""
        
        readme_path = class_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Created directory: {class_dir}")
    
    # Create main dataset README
    main_readme_content = """# Human Classifier Dataset

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
"""
    
    main_readme_path = dataset_dir / "README.md"
    with open(main_readme_path, 'w') as f:
        f.write(main_readme_content)
    
    print(f"✅ Created main README: {main_readme_path}")
    print("\n🎯 Next Steps:")
    print("1. Add 10-20 video files to each class directory")
    print("2. Remove the README.md files from class directories")
    print("3. Run: python src/model/train.py")
    print("4. Start the API: python src/api/main.py")
    print("5. Start the frontend: cd src/frontend && npm start")


if __name__ == "__main__":
    print("🚀 Creating sample dataset structure...")
    create_sample_dataset()
    print("\n✨ Dataset structure created successfully!")
