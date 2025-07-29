# Image Forgery Detection using CNN, MobileNetV2 and Streamlit

This project detects forged (tampered) images using two deep learning models:
- A custom Convolutional Neural Network (CNN)
- A pre-trained MobileNetV2 model using transfer learning

The project also includes a simple Streamlit interface that allows users to upload an image and get instant predictions.

---

## 🧠 Project Overview

This project aims to classify images into:
- **Authentic (Original)**
- **Forged (Tampered)**

We trained two models for this task:
1. A CNN built from scratch using Keras.
2. A fine-tuned MobileNetV2 model with a custom classification head.

---

## 🧪 Models Used

### 🔹 Custom CNN
- 3 Convolutional layers
- ReLU activations
- MaxPooling
- Dense + Dropout
- Output: Softmax layer with 2 units

### 🔹 MobileNetV2 (Transfer Learning)
- Pretrained base (frozen)
- Custom head with GlobalAveragePooling + Dense layers
- Output: 2-class softmax

---

## 🎯 Results

- **CNN Accuracy**: ~63%  
- **MobileNetV2 Accuracy**: ~71% 

---

## 🖼️ Streamlit App

Upload an image → Get prediction (Original or Forged) + Confidence scores

To run:

```bash
streamlit run app.py
