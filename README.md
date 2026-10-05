# 🔍 AI Image Authenticity Checker & Deepfake Detector

An end-to-end, explainable AI (XAI) deep learning system designed to analyze uploaded images, predict whether they are **REAL** or **MANIPULATED / DEEPFAKE**, compute class confidence scores, and render interactive **Grad-CAM** visual heatmaps pointing out suspicious image regions.

---

## 📌 Project Overview

Digital image manipulation and AI-generated deepfakes present significant challenges to media integrity. This repository provides a modular, production-ready computer vision pipeline built with **PyTorch**, **EfficientNet-B0**, **Grad-CAM**, and **Streamlit**.

### ⚠️ Forensic Disclaimer
> **Important Notice:** This application serves as an **AI-driven image manipulation screening tool**. Predictions and Grad-CAM visual heatmaps are generated for investigative assistance and decision support. The system **does not provide absolute or legally binding digital forensic proof** of authenticity.

---

## ✨ Features

- **Transfer Learning Backbone:** Fine-tuned **EfficientNet-B0** pretrained on ImageNet for high feature extraction capacity with fast inference speed.
- **Robust Preprocessing & Augmentation:** ImageNet normalization, random rotations, flips, color jittering, and graceful error handling for corrupted/invalid images.
- **Explainable AI (Grad-CAM):** Visualizes exact pixel regions that influenced the model's decision using backpropagated gradient heatmaps overlaid onto the original image.
- **Comprehensive Model Evaluation:** Automated evaluation calculating Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC curve plots.
- **Professional Streamlit Web App:** Interactive web application with real-time analysis, side-by-side visualizers, confidence meters, and synthetic test dataset generator.
- **Cross-Platform & Hardware Adaptive:** Automatically detects CUDA GPU hardware when available and defaults to CPU execution without configuration changes.

---

## 🏗️ Project Architecture

```text
deepfake detector/
├── dataset/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── validation/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── models/
│   ├── .gitkeep
│   └── deepfake_detector.pth      # Saved model checkpoint after training
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Transforms, custom dataset & dataloaders
│   ├── train.py                   # Transfer learning loop & validation monitor
│   ├── evaluate.py                # Metrics evaluation (Acc, F1, ROC-AUC, CM)
│   ├── predict.py                 # Single image inference & confidence scoring
│   ├── explain.py                 # Grad-CAM XAI heatmap generation & overlay
│   └── utils.py                   # Hardware detection, seeds, synthetic dataset
├── app.py                         # Streamlit UI application
├── requirements.txt               # Dependencies list
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Deep Learning Framework:** PyTorch & Torchvision
- **Model Architecture:** EfficientNet-B0
- **Explainable AI:** Grad-CAM (Custom Gradient-based Class Activation Mapping)
- **Computer Vision & Image Processing:** OpenCV, Pillow (PIL)
- **Data & Metrics:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Web UI:** Streamlit

---

## 🚀 Getting Started & Installation

### 1. Clone or Open Workspace
Ensure you are in the project root directory:
```bash
cd "deepfake detector"
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

Organize your dataset inside `dataset/` following this structure:

```text
dataset/
├── train/
│   ├── real/    (Clean, unmanipulated images)
│   └── fake/    (Deepfake or manipulated images)
├── validation/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### 🎨 Quick Demo / Synthetic Dataset Generator
If you do not have a custom dataset ready, you can automatically generate a synthetic benchmark dataset with 1-click:
```bash
python -m src.utils
```
This generates synthetic real and fake images inside `dataset/` for instant testing and pipeline verification.

---

## 🏋️ Training Instructions

Train the EfficientNet-B0 model using the CLI:

```bash
python -m src.train --epochs 10 --batch_size 16 --lr 0.0001
```

### Key Training Features:
- Automatically uses GPU (`cuda`) if available, else defaults to `cpu`.
- Monitors validation loss/accuracy after every epoch.
- Saves the best checkpoint to `models/deepfake_detector.pth`.
- Saves training loss/acc history to `models/training_history.json`.

---

## 📊 Evaluation Instructions

Evaluate the trained model on your test or validation set:

```bash
python -m src.evaluate --model_path models/deepfake_detector.pth --split test
```

### Output Artifacts:
- Metrics printed to terminal (Accuracy, Precision, Recall, F1, ROC-AUC).
- Metrics saved to `evaluation_results/metrics.json` and `evaluation_results/metrics_summary.txt`.
- Visual plots saved:
  - `evaluation_results/confusion_matrix.png`
  - `evaluation_results/roc_curve.png`

---

## 🎯 Command-Line Prediction & Grad-CAM

Run prediction on a single image file via terminal:

```bash
python -m src.predict --image_path path/to/sample.jpg
```

Generate and save a Grad-CAM heatmap visualization for an image:

```bash
python -m src.explain --image_path path/to/sample.jpg --output_path grad_cam_output.png
```

---

## 🌐 Launching the Streamlit Web Application

Launch the interactive web UI:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### App Features:
1. **Image Upload:** Drag and drop any JPG, JPEG, PNG, or WEBP image.
2. **Real-time Prediction Card:** Displays `REAL` or `MANIPULATED` badge with percentage confidence.
3. **Probability Distribution:** Visual bars for class probabilities.
4. **Grad-CAM Explainable AI:** Tabbed visualizer showing Original Image, Grad-CAM Overlay, Raw Heatmap, and Side-by-Side comparison.
5. **Sidebar Quick Tools:** Generate synthetic data and trigger training directly from the UI.

---

## 🔄 Example Workflow

1. **Install requirements:** `pip install -r requirements.txt`
2. **Generate demo data:** `python -m src.utils`
3. **Train detector model:** `python -m src.train --epochs 5`
4. **Evaluate model:** `python -m src.evaluate`
5. **Launch Streamlit app:** `streamlit run app.py`

---

## ⚠️ Limitations & Future Improvements

### Current Limitations:
- The synthetic generator creates basic geometric noise artifacts for demonstration; real-world deepfakes require training on authentic benchmarks (e.g., FaceForensics++, Celeb-DF, DeepFake Detection Challenge).
- Facial detection/cropping preprocessing is not included by default to allow generic image manipulation screening.

### Future Improvements:
- Integrate MTCNN or RetinaFace for automatic face extraction prior to classification.
- Support frequency domain analysis (FFT / DCT high-frequency residual analysis).
- Add support for video file deepfake detection by processing frame sequences.
