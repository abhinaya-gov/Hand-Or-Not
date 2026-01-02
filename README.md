<p align="center">
  <a href="#model-architecture">
    <img src="https://img.shields.io/badge/Architecture-View-blue?style=for-the-badge" />
  </a>
  <a href="#datasets">
    <img src="https://img.shields.io/badge/Datasets-Explore-green?style=for-the-badge" />
  </a>
  <a href="#future-enhancements">
    <img src="https://img.shields.io/badge/Future%20Plans-Roadmap-orange?style=for-the-badge" />
  </a>
</p>

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

## Datasets

This project uses a combination of real and synthetic hand image datasets to train and evaluate a **binary hand detection model (hand vs no-hand)**.

### Hand-Bo3ot (Roboflow Universe)
A general-purpose hand detection dataset with bounding box annotations across different poses, lighting conditions, and backgrounds. Used as the primary dataset for training the detector.  
https://universe.roboflow.com/yolov4tiny-wzb2k/hand-bo3ot

### Bharatanatyam Mudras (Roboflow Universe)
Contains annotated images of classical Indian hand gestures (mudras). Used to add diversity in hand shapes, orientations, and fine-grained poses.  
https://universe.roboflow.com/mudras-avdrb/bharatanatyam-mudras-fg9qo-gcruc

### Hand Gesture Dataset (Roboflow Universe)
A collection of hand gesture images used for additional data exploration and optional augmentation.  
https://universe.roboflow.com/horyzn-qhfq4/hand-gesture-gizg2

### Hand Detection Dataset — VOC/YOLO Format (Kaggle)
A ready-to-use dataset in VOC/YOLO format for standard object detection training and benchmarking.  
https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format/data

### Synthetic Hand Detection Dataset (Kaggle)
A synthetic dataset used to improve generalization and robustness, especially for rare poses and edge cases.  
https://www.kaggle.com/datasets/zeyadkhalid/hand-detection

These datasets together provide a mix of real-world variability and synthetic augmentation, helping the model generalize better across different environments, poses, and lighting conditions.


## Model Architecture

**HandOrNotCNN**:
- 4 convolutional blocks (3→32→64→128→128 channels)
- Each block: Conv2d → BatchNorm → ReLU → MaxPool
- Fully connected layers: 128×7×7 → 64 → 32 → 2
- Dropout (0.5) for regularization

<img width="1978" height="854" alt="image" src="https://github.com/user-attachments/assets/c1f403b6-2f3a-445a-bbf6-a2208a12d602" />


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
- **Training Accuracy**: ~96%
- **Dev Accuracy**: ~98%
- **Precision**: ~97%
- **Recall**: ~99%

## Future Enhancements

Possible improvements and extensions for this project include:

- **Temporal modeling for video streams**  
  Add temporal smoothing or sequence models (e.g., moving average, LSTM, or 3D CNN) to improve stability in video inference.

- **Domain adaptation and robustness**  
  Improve generalization to:
  - Different skin tones
  - Gloves, accessories, and occlusions
  - Extreme lighting or low-quality cameras

- **Active learning loop**  
  Allow users to correct predictions and continuously improve the model with new labeled data.

- **Explainability and debugging tools**  
  Add Grad-CAM or feature visualization to better understand what the model focuses on during prediction.

- **Benchmark against pretrained models**  
  Compare performance with pretrained backbones like MobileNet, ResNet, or EfficientNet.

These enhancements would evolve the project from a learning-focused prototype into a more robust, scalable, and production-ready vision system.


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
