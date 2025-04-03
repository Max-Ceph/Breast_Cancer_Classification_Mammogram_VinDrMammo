# Breast_Cancer_Classification_Mammogram_VinDrMammo

## 📌 Project Overview

This project was completed as part of the **Statistics and Data Science (Final Project)** course. It presents a binary classification pipeline for **mammogram images** using traditional machine learning models. The task is to distinguish **benign (normal)** from **malignant (abnormal)** cases using the **VinDr-Mammo** dataset.

---

## 🧠 Objective

The goal is to predict whether a mammogram is **normal (BI-RADS 1)** or shows **abnormalities (BI-RADS 2–5)**. This was treated as a **binary classification** problem using multiple feature extraction techniques and machine learning models.

---

## 📊 Dataset

- **Name**: VinDr-Mammo (VinBigData Institute)
- **Labels**:  
  - `Class 0`: Normal (BI-RADS 1)  
  - `Class 1`: Abnormal (BI-RADS 2, 3, 4, 5)

- **Link**: https://vindr.ai/datasets/mammo

---

## 🛠️ Feature Extraction Techniques

1. **Raw Pixel Features**  
   - Grayscale conversion, resized to 128×128  
   - Normalized to [0, 1]

2. **HOG (Histogram of Oriented Gradients)**  
   - Extracted from 224×224 resized images  
   - Captures edge/orientation information

3. **ResNet50 Features**  
   - Deep features extracted using pretrained ResNet50  
   - Based on convolutional activations

---

## ⚙️ Data Preprocessing

- Handled missing or unreadable images with skip logic
- Applied label transformation (BI-RADS → binary)
- Used `StandardScaler` to normalize feature values
- Balanced data using:
  - `class_weight='balanced'` (for LR, SVM)
  - Downsampling (for RF, DT)
- Limited training size for SVM (1000 train / 200 test) due to time complexity
- Saved processed features as `.npy` for reuse

---

## 🤖 Models Implemented (Scikit-learn)

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- k-Nearest Neighbors (k-NN)

---

## 📈 Evaluation Metrics

Each model was assessed using:
- **Accuracy**
- **Precision**
- **Recall** (priority in medical settings)
- **F1-score**
- **ROC-AUC**

---

## ✅ Key Findings

- **Best recall**: Random Forest & SVM  
- **Best speed–recall trade-off**: k-NN  
- **Fastest**: Logistic Regression  
- **Least reliable (low recall)**: Decision Tree  
- **Final verdict**:  
  - *k-NN* was chosen as the best overall model  
  - *SVM* and *Random Forest* showed strongest performance, but were computationally expensive

---

