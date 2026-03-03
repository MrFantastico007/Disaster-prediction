# 🌪️ Multi-Task Disaster & Severity Classification

A deep learning computer vision pipeline built with TensorFlow/Keras that simultaneously predicts the **type of a natural disaster** and its **intensity level** from a single image using a multi-output architecture.

## 🧠 Architecture Overview

This model utilizes a **Multi-Task Convolutional Neural Network (CNN)**. It leverages Transfer Learning from a pre-trained `EfficientNetB0` backbone to extract high-level features, which are then passed into two separate prediction heads.

* **Input Image:** 224x224 RGB
* **Augmentation:** Random Horizontal Flip, 30% Rotation, and 20% Contrast variation.
* **Feature Extractor:** `EfficientNetB0` (Pre-trained on ImageNet).
* **Pooling:** `GlobalAveragePooling2D` + `BatchNormalization`.

### 🔀 The "Y-Split" Branches
1. **Disaster Type Head:** Classifies the image into `Earthquake`, `Flood`, or `Wildfire`.
    * `Dense(256)` ➡️ `Dropout(0.5)` ➡️ `Dense(3, Softmax)`
2. **Severity Head:** Classifies the subjective intensity level into `High`, `Medium`, or `Low`. It uses a deeper architecture and an increased loss weight (2.5x) to handle the complexity and subjectivity of this task.
    * `Dense(512)` ➡️ `Dropout(0.4)` ➡️ `Dense(256)` ➡️ `Dropout(0.4)` ➡️ `Dense(3, Softmax)`

## 🎯 Model Training Methodology (Transfer Learning)
The network was trained in two distinct phases:
* **Phase 1 (Feature Extraction):** The entire EfficientNet baseline is frozen. The new classification and severity heads are trained for 15 epochs.
* **Phase 2 (Fine-Tuning):** The top 30 layers (the last two computational blocks) of EfficientNet are unfrozen and trained simultaneously with the heads at an ultra-low learning rate (`1e-5`) for another 30 epochs.

*Note: `EarlyStopping` (patience=5) and `ReduceLROnPlateau` callbacks ensure optimal weight restoration and dynamically prevent overfitting.*

## 📊 Evaluation Results (Test Set)
* **Disaster Type Accuracy:** `91.57%`
* **Severity Intensity Accuracy:** `36.00%` (subjective visual analysis limitation)
* *The model is vastly superior at detecting the class of disaster, effectively separating Earthquakes, Floods, and Wildfires with minimal confusion.*

## 📂 Dataset Structure
If retraining the model on your own dataset, structure your images perfectly inside the `disaster_final` folder like this:
```text
disaster_final/
├── train/
│   ├── earthquake/
│   │   ├── high/
│   │   ├── medium/
│   │   └── low/
│   ├── flood/
│   ...
├── validation/
└── test/
