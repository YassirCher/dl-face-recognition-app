# 🤖 Deep Learning: Advanced Face Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv&logoColor=white)](https://opencv.org/)

## 📝 Project Context
This project was developed as part of the **Deep Learning** module. It represents a comprehensive exploration of face recognition architectures, focusing on the trade-offs between various backbones and classifier heads.

## 🚀 Project Overview
We conducted an extensive benchmarking study, training and evaluating **90 different models**. This was achieved by combining multiple state-of-the-art backbones (VGG, ResNet, EfficientNet, ConvNeXt, ViT, Swin Transformer) with various classical and neural classifier heads.

- **Scale:** 90 unique model combinations.
- **Effort:** Spanning **3 detailed notebooks**.
- **Training Time:** Approximately **30 hours** of total compute time.
- **Outcome:** A functional real-time inference application and a deep-dive analytics dashboard.

## 📸 App Showcase

### 🔍 Live Inference Page
The inference engine features real-time face detection and classification with live FPS and latency monitoring.
![Inference App](screenshots/inference_app.png)

### 📊 Comprehensive Analysis Dashboard
The integrated dashboard provides detailed metrics, performance heatmaps, and efficiency frontier visualizations for all 90 models.
![Analysis Page](screenshots/analysis_page.png)

## ✨ Features
- **Real-time Inference:** Process live video or uploaded files with on-the-fly model switching.
- **Extensive Benchmarking:** Compare 90 models across Accuracy, F1-Score, AUC-ROC, and Inference Speed.
- **Deep Analytics:** Visualization of training health, backbone stability, and model rankings.
- **Modern UI:** A premium, dark-mode interface built with glassmorphism and real-time Chart.js integration.

## 👥 Authors
- **CHERGUI Yassir**
- **ZOUITNI Salaheddine**

---

## 🛠️ How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python app.py
```
Then open your browser and navigate to:
- **Main App:** `http://localhost:5000`
- **Analytics:** `http://localhost:5000/dashboard`

---
*Developed for the Deep Learning Module - 2026*
