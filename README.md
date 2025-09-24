# Biologically Inspired CNN for MNIST Digit Classification

## Overview
This project implements a **biologically inspired convolutional neural network (CNN)** to classify handwritten digits from the MNIST dataset. 
It integrates **ON/OFF retinal filters** and **edge-detection kernels** to mimic early visual processing in the human eye, combined with trainable neural layers for classification. 
A key highlight is the **visualization of intermediate feature maps** to interpret how the network processes digits.

## Features
- **Retina-inspired feature extraction:** ON-center and OFF-center filters detect contrast and salient features.
- **Edge detection and intersections:** Horizontal, vertical, and diagonal edge kernels, including combined intersections, capture complex patterns.
- **Hybrid CNN architecture:** Combines handcrafted filters with trainable Dense layers.
- **Feature visualization:** Visualizes retina, edge, and intersection activations to enhance interpretability.
- **Performance:** Achieved **97.29% test accuracy** on MNIST with confusion matrix evaluation.

## Technical Details
- **Language & Framework:** Python, TensorFlow/Keras, NumPy, Matplotlib, Seaborn
- **Input:** 28x28 grayscale MNIST images
- **Model Architecture:**
  1. Retina layer: 2-channel convolution with ON/OFF filters
  2. Thresholding + Batch Normalization
  3. Edge layer: 4-channel convolution (horizontal, vertical, diagonal)
  4. Intersection features via element-wise multiplication
  5. Concatenate all features
  6. Flatten → Dense(256, ReLU) → Dropout(0.35) → Dense(10, Softmax)
- **Training:**
  - Optimizer: Adam
  - Loss: Sparse Categorical Crossentropy
  - Epochs: 35
  - Batch size: 350

## Visualization
- **Retina outputs:** ON-center and OFF-center activations
- **Edge maps:** Horizontal, vertical, diagonal 45°, diagonal 135°
- **Intersections:** Combined edge activations for complex feature patterns

All visualizations use heatmaps for intuitive interpretation.

## How to Run
1. Install dependencies:
```bash
pip install numpy tensorflow matplotlib seaborn scikit-learn
