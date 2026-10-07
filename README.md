# DeepFER: Facial Emotion Recognition Using Deep Learning

DeepFER is an individual deep-learning project that classifies facial
expressions into seven categories using image classification and transfer
learning.

## Author

**Dolly Jain**

## Project Overview

The objective of this project is to develop and evaluate a deep-learning
system capable of recognizing facial-expression categories from images.

The supported expression classes are:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

The project includes dataset validation, exploratory data analysis, duplicate
detection, data preprocessing, model comparison, fine-tuning, final test
evaluation, model interpretability, image-upload inference, and a webcam
face-capture demonstration.

## Problem Statement

Facial-expression recognition is a challenging computer-vision problem because
facial appearance changes with lighting, camera quality, head position, image
resolution, expression intensity, and individual facial characteristics.

The goal of this project is to train a multiclass image-classification model
and evaluate how reliably it distinguishes seven facial-expression categories.

## Dataset

The source dataset contains grayscale facial images organised into seven
expression classes.

During data preparation, the project performs:

- Dataset archive inspection
- Image decoding and validation
- Duplicate-directory verification
- Exact duplicate-image detection
- Conflicting-label detection
- Cross-split duplicate removal
- Class-distribution analysis
- Brightness and contrast analysis
- Edge-strength analysis
- Stratified training, validation, and test splitting

After cleaning and splitting, the project uses:

| Dataset subset | Images |
|---|---:|
| Training | 21,609 |
| Validation | 5,403 |
| Test | 6,965 |
| Total | 33,977 |

## Exploratory Data Analysis

The notebook investigates:

- Class imbalance
- Image dimensions and colour modes
- Invalid or unreadable images
- Duplicate and conflicting images
- Class-level brightness differences
- Pixel-intensity variability
- Edge-strength distributions
- Similarity between emotion classes
- PCA-based visual analysis
- Statistical relationships between dataset characteristics

The analysis shows that several expression classes have visually similar
patterns, making some classifications difficult.

## Data Preprocessing

The preprocessing pipeline includes:

- Image decoding
- Resizing
- Pixel normalization
- Grayscale-compatible three-channel conversion
- Data augmentation
- Stratified dataset splitting
- Class-weight calculation

For webcam inference, the system additionally performs:

- Haar-cascade face detection
- Central face selection
- Face cropping
- Grayscale conversion
- Histogram equalization
- Confidence and uncertainty analysis

## Models

The project evaluates deep-learning approaches including:

- Baseline Convolutional Neural Network
- MobileNetV2 transfer-learning model
- Fine-tuned MobileNetV2 model

The final selected model is:

```text
mobilenetv2_fine_tuned_continued_best.keras
