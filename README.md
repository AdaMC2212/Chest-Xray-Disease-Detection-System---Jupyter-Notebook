# Chest-Xray-Disease-Detection-System

## Project Overview

This project applies deep learning to classify chest X-ray images into three categories: **Normal**, **Pneumonia**, and **Tuberculosis**. It involves designing and optimizing both **Convolutional Neural Networks (CNN)** and **hybrid CNN-RNN models**, supported by preprocessing, augmentation, and hyperparameter tuning strategies.

## Motivation

Pneumonia and tuberculosis are serious respiratory diseases that require early diagnosis for effective treatment. Deep learning offers the potential to assist radiologists in automating diagnosis using medical imaging, especially in under-resourced environments.

---

## Dataset

- **Source**: [Kaggle – Combined Pneumonia, Tuberculosis & Normal X-ray](https://www.kaggle.com/datasets/rifatulmajumder23/combined-unknown-pneumonia-and-tuberculosis)
- **Classes**: 
  - `Normal`: Healthy lungs
  - `Pneumonia`
  - `Tuberculosis`
  - `Unknown` (removed for training)

## Preprocessing

- Removed all `UNKNOWN` labeled images.
- Resized images to **128×128**.
- Converted to **grayscale** format.
- Rescaled pixel values to `[0, 1]` using `ImageDataGenerator(rescale=1./255)`.
- Applied data augmentation (rotation, zoom, shift) on training set.
- Computed **class weights** to address imbalance in label distribution.

---

## Exploratory Data Analysis (EDA)

- Visualized:
  - Class distributions per split
  - Sample X-ray images by class
  - Image dimension consistency
- Graphed training/validation accuracy & loss curves
- Checked for duplicates and missing metadata

---

## Model Architectures

### 1️) Base CNN
- 3 × Conv2D layers with MaxPooling
- Flatten → Dense(128) → Dropout(0.3)
- Output: Dense(3, softmax)

### 2️) Tuned CNN (via Keras Tuner)
- Tuned:
  - Filters, kernel size, dense units, dropout
- 1 × `BatchNormalization` layer after first Conv2D
- Optimizer: Adam (`lr=0.0005`)

### 3️) Base CNN-RNN
- CNN for spatial feature extraction
- Reshape → RNN (LSTM or GRU)
- Output: Dense(3, softmax)
- Captures sequential spatial patterns in image regions

### 4️) Tuned CNN-RNN
- Tuned similar hyperparameters + dropout
- Used RNN layer (LSTM) after Conv blocks
- Focused on reducing overfitting and boosting recall

---

## Evaluation Metrics

- Accuracy
- Loss
- Confusion Matrix
- Precision, Recall, F1-score per class

---

## Results Summary

| Model           | Accuracy | Notes                                 |
|----------------|----------|----------------------------------------|
| Base CNN        | ~96%     | Stable and good generalization         |
| Tuned CNN       | ~96%     | Slightly improved recall and F1        |
| Base CNN-RNN    | ~96%     | Better at distinguishing Pneumonia/TB  |
| Tuned CNN-RNN   | ~97%     | Highest overall performance            |

---

