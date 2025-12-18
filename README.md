# Hand-Or-Not
A convolutional neural network trained to distinguish between hand and non-hand images, enabling reliable hand presence detection in computer vision tasks.

# Hand Detection CNN

A PyTorch-based Convolutional Neural Network for real-time hand detection from webcam feeds. The model classifies images as either containing a hand or not containing a hand.

## Overview

This project trains a custom CNN to distinguish between images with hands and images without hands, then deploys the model for real-time inference using a webcam.

## Features

- Custom CNN architecture with 4 convolutional blocks
- Data augmentation for improved generalization
- Real-time webcam inference
- Batch normalization and dropout for regularization
- Performance metrics including precision, recall, and confusion matrix

## Requirements

```
torch
torchvision
numpy
seaborn
matplotlib
pandas
opencv-python (cv2)
pillow
scikit-learn
```

Install dependencies:
```bash
pip install torch torchvision numpy seaborn matplotlib pandas opencv-python pillow scikit-learn
```

## Dataset Structure

The code expects the following directory structure:

```
hands/
├── dataset1/
│   ├── train/
│   │   └── images/
│   │       ├── *.jpg
│   │       └── *.png
│   └── test/
│       └── images/
│           ├── *.jpg
│           └── *.png
├── dataset2/
│   └── ...
└── ...

Negatives/
├── dataset1/
│   └── images/
│       ├── *.jpg
│       └── *.png
│   └── NoObject/
│       └── *.png
└── ...
```

- `hands/`: Contains datasets with hand images
- `Negatives/`: Contains datasets without hand images

## Model Architecture

**HandOrNotCNN**:
- 4 convolutional blocks (3→32→64→128→128 channels)
- Each block: Conv2d → BatchNorm → ReLU → MaxPool
- Fully connected layers: 128×7×7 → 64 → 32 → 2
- Dropout (0.5) for regularization

**Input**: 224×224 RGB images  
**Output**: Binary classification (hand/no_hand)

## Training

The model uses:
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Data Augmentation**: Random horizontal flip, rotation, color jitter
- **Normalization**: ImageNet mean/std
- **Train/Dev Split**: 90/10 for negative samples

Run training:
```python
python train.py  # or run the notebook cells
```

Training outputs:
- Epoch-wise train/dev loss and accuracy
- Loss history plot
- Saved model: `hand_model.pth`

## Evaluation

The model is evaluated on:
- **Accuracy**: Overall classification accuracy
- **Precision**: Correct hand predictions / total hand predictions
- **Recall**: Correct hand predictions / actual hands
- **Confusion Matrix**: Visualization of true/false positives/negatives

## Real-Time Inference

Run the webcam classifier:

```python
python webcam_inference.py  # or run the relevant notebook cells
```

**Controls**:
- Press `q` to quit

The script:
1. Loads the trained model (`better_hand_class.pth`)
2. Opens the default webcam
3. Processes each frame through the model
4. Displays prediction and confidence on screen

### Configuration

Edit these variables in the inference section:

```python
MODEL_PATH = "better_hand_class.pth"  # Path to saved model
CLASS_NAMES = ["no_hand", "hand"]     # Class labels
INPUT_SIZE = 224                      # Input image size
USE_GPU = torch.mps.is_available()    # Enable GPU (MPS for Mac, CUDA for others)
```

## Performance

Metrics:
- **Training Accuracy**: ~95%
- **Dev Accuracy**: ~92%
- **Precision**: ~93%
- **Recall**: ~91%

## File Outputs

- `hand_model.pth` / `better_hand_class.pth`: Trained model weights
- `confusion_matrix.png`: Confusion matrix visualization

## Notes

- The code uses MPS (Metal Performance Shaders) for Mac GPU acceleration
- For NVIDIA GPUs, change device detection to use CUDA
- Adjust batch size based on available memory
- The 90/10 train/dev split is applied only to negative samples (dev ratio = 0.1)

## Troubleshooting

**Camera not opening**: 
- Check if another application is using the webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` for external cameras

**Out of memory errors**:
- Reduce batch size (default: 32)
- Reduce image size (default: 224×224)

**Low accuracy**:
- Increase training epochs
- Add more diverse training data
- Adjust learning rate or augmentation parameters

## License

This project is intended for educational purposes. A license can be added later if the repository is extended or shared for reuse.

## Author

Abhi Learning-focused Machine Learning & Python projects
