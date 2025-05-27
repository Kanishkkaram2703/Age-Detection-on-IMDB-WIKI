# Age-Detection-on-IMDB-WIKI

## 1. Abstract

This project implements a deep learning-based age detection model trained on the IMDB-WIKI dataset. It leverages Convolutional Neural Networks (CNNs) to predict the age of individuals based on facial images. The model aims to automate age estimation, which can be useful in demographic analysis, content restriction systems, and smart security solutions.

## 2. Introduction

### Problem Statement:
Accurate age prediction from images is a challenging computer vision task. Traditional approaches using hand-crafted features fail to generalize well across diverse datasets. This project uses CNN-based deep learning to build an effective and scalable age detection model.

### Objectives:

- Build a CNN model from scratch to predict age from facial images.
- Preprocess and utilize the IMDB-WIKI dataset effectively.
- Evaluate model performance using metrics such as MAE (Mean Absolute Error).
- (Optional) Prepare the system for real-time or image-based age inference.

### Applications:

- Digital content control and age-restricted platforms.
- Demographic analysis and marketing.
- Access control and identity verification systems.

## 3. Literature Review

- **IMDB-WIKI Dataset**: One of the largest public face datasets with age and gender labels.
- **CNNs for Regression**: Deep CNNs are preferred over traditional machine learning for image-based regression tasks due to their feature learning capabilities.
- **Preprocessing Importance**: Face cropping, grayscale conversion, and normalization significantly influence model accuracy.

## 4. Methodology

### Data Collection:
- The dataset used is the **IMDB-WIKI** face dataset, consisting of over 500k images with age and gender labels.

### Preprocessing:
- **Grayscale conversion** for simplicity.
- **Resizing** images to a consistent shape (e.g., 48x48 or 96x96).
- **Face detection** to crop only relevant regions.
- **Normalization** of pixel values to [0, 1].

### Model Training:
- **Model**: A custom-built CNN with Conv2D, MaxPooling, Flatten, and Dense layers.
- **Loss Function**: Mean Squared Error (MSE) for regression.
- **Optimizer**: Adam.
- **Callbacks**: EarlyStopping and ReduceLROnPlateau to prevent overfitting.

### Evaluation:
- Accuracy and Mean Absolute Error (MAE).
- Plots of loss vs. epochs.

## 5. Implementation

### Core Modules:
- `model_training.ipynb`: Contains preprocessing, training, and evaluation code.
- `utils.py` (optional): Utility functions for image loading, processing, etc.

### Sample Preprocessing Code:

```python
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    return img.reshape(96, 96, 1)

**## CNN Architecture:**
python
Copy code
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(96,96,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)  # Regression output
])
```

**## 6. Results**
Performance Metrics:
Training MAE: ~5–7 years

Validation MAE: ~6–8 years

Plots:
Include graphs showing training/validation loss and MAE trends.

**7. Challenges and Limitations**
Challenges:
Handling extreme age ranges (1–100+ years).

Dataset imbalance (more images for young adults).

Noisy labels in the IMDB-WIKI dataset.

Limitations:
Real-time inference not implemented in this version.

Model may mispredict for outliers or poor-quality images.

**8. Conclusion**
This project successfully demonstrates a custom deep learning pipeline for age detection using the IMDB-WIKI dataset. With improved preprocessing and model tuning, real-time applications and multi-label outputs (age + gender) can be explored.

**9. References**
IMDB-WIKI Dataset on Kaggle

Keras Documentation

OpenCV Face Detection

TensorFlow Guide

**🚀 Features**
Age detection using grayscale CNNs

Trained on the IMDB-WIKI dataset

MAE-based regression output

Lightweight and easy to train

**📁 Dataset**
IMDB-WIKI Dataset: A public dataset containing over 500k face images with age and gender labels.

Not included in this repo due to size.

Download from Kaggle IMDB-WIKI dataset

**✅ Requirements**
Python 3.8 or higher

OpenCV

NumPy

TensorFlow/Keras

Matplotlib

Scikit-learn

Pillow

**💾 Installation**

git clone https://github.com/Kanishkkaram2703/Age-Detection-on-IMDB-WIKI.git
cd Age-Detection-on-IMDB-WIKI
pip install -r requirements.txt
🧠 Model Training
To train the model:

Download and place the dataset in a folder named dataset/.

Open model_training.ipynb.

Run all the cells step by step to preprocess, train, and evaluate the model.

The trained model will be saved as .keras.

**🔍 Usage**
(Optional GUI or live feed detection not included by default.)

To use the trained model:

Load any image using OpenCV or PIL

Resize and normalize it

Pass it to the model for prediction

## 🔄 Pre-trained Model

Download the trained `.keras` model directly from this Google Drive folder:  
📦 [Google Drive - Age Detection Model](https://drive.google.com/drive/folders/1qsjD8CMuT5IU3eQ2XtIxaIE2TESI6iet?usp=drive_link)

