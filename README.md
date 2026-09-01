---
title: EmoVision AI Pro - Real-Time Facial Emotion Detection
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.62.0"
app_file: app.py
pinned: false
license: mit
---

# Facial Emotion Detection using Deep Learning (v2.0)

A deep learning facial expression recognition system capable of classifying human facial expressions into **seven emotions** (`Angry`, `Disgusted`, `Fearful`, `Happy`, `Neutral`, `Sad`, `Surprised`) in real time from live webcam feeds, uploaded photos, or the **FER-2013** dataset.

![Accuracy Plot](accuracy.png)

---

## 🌟 Key Features

- **Normalized Real-Time Inference**: Fixed input normalization discrepancy between training and live inference pipelines.
- **Robust Face Detection**: High-accuracy **MediaPipe Face Detection** (BlazeFace SSD topology) with automatic offline fallback to classical **OpenCV Haar Cascades** (`haarcascade_frontalface_default.xml`).
- **Modernized Deep Learning Stack**: Full compatibility with **TensorFlow 2.x / Keras 3.x**, Python 3.10–3.12, and Apple Silicon (`arm64`) as well as `x86_64`.
- **Architectures**:
  - **4-Block Custom CNN**: Lightweight model (~63.2% test accuracy on FER-2013, 2.66 ms latency / 376 FPS).
  - **MobileNetV3-Small Baseline**: Transfer-learning baseline (~65.8% test accuracy, 23.64 ms latency / 42 FPS) with data augmentation layers.
- **Interactive Web UI**: Modern dark-themed **Streamlit application** with multi-face detection, photo upload analysis, snapshot camera mode, and per-class probability bar charts.
- **Automated Test Suite**: Full `pytest` verification covering dataset preprocessing, model architectures, face detection fallbacks, and end-to-end training regression.

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Decoder420/Emotion-Detection-using-Facial-Recognition-.git
   cd Emotion-Detection-using-Facial-Recognition-
   ```

2. **Create a virtual environment (Python 3.10–3.12 recommended):**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install pinned dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Data Preparation (FER-2013)
Download `fer2013.csv` from [Kaggle](https://www.kaggle.com/deadskull7/fer2013) into the project root, then run:
```bash
python dataset_prepare.py --csv ./fer2013.csv --output data
```

### 2. Model Training
Train the CNN or MobileNetV3 model with data augmentation and modern Keras callbacks:
```bash
# Train standard 4-block CNN
python emotions.py --mode train --model cnn --epochs 50 --augment

# Train MobileNetV3-Small
python emotions.py --mode train --model mobilenet --epochs 50 --augment
```

### 3. Continuous Live Webcam Feed (OpenCV GUI)
Real-time 60 FPS continuous video inference with MediaPipe or Haar Cascade:
```bash
python emotions.py --mode display --detector mediapipe
```

### 4. Interactive Web Application (Streamlit)
Launch the Streamlit web dashboard for photo uploads and multi-face emotion confidence charts:
```bash
streamlit run app.py
```
> [!NOTE]
> The Streamlit web app utilizes `st.camera_input` which operates in **snapshot mode** (capturing single frames on demand). For **continuous 60 FPS live video streaming**, use the CLI tool (`python emotions.py --mode display`).

### 5. Benchmark Architectures
Compare FLOPs, parameters, inference latency (ms), throughput (FPS), and test accuracy:
```bash
python src/benchmark.py
```

| Model Architecture | Parameters | FER-2013 Accuracy | Avg Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **4-Block Custom CNN** | 2,345,607 | 63.2% | **2.66 ms** | **376 FPS** |
| **MobileNetV3-Small (Transfer)** | 1,088,637 | **65.8%** | 23.64 ms | 42 FPS |

### 6. Run Test Suite
Run the automated test suite with pytest:
```bash
pytest -v
```
> [!NOTE]
> The 11-test automated suite covers dataset parsing, model architectures, face detection fallback paths, unified inference, and training regression. The Streamlit UI layer (`app.py`) is verified manually in the browser.

---

## 📁 Repository Structure

```
Emotion-Detection-using-Facial-Recognition-/
├── app.py                                 # Streamlit web application (manually verified UI)
├── dataset_prepare.py                     # FER-2013 CSV parser & image generator
├── emotions.py                            # Main CLI training & webcam display script
├── haarcascade_frontalface_default.xml    # Permanent offline Haar Cascade fallback asset
├── requirements.txt                       # Pinned dependencies for Python 3.10-3.12
├── CHANGELOG.md                           # Detailed record of fixes & modernization
├── MIGRATION.md                           # Guide for upgrading from v1.0
├── src/
│   ├── __init__.py
│   ├── models.py                          # Model definitions & unified preprocessing
│   ├── face_detector.py                   # MediaPipe & Haar Cascade face detection
│   ├── inference.py                       # Unified emotion detection engine
│   └── benchmark.py                       # Latency, parameter & accuracy benchmarking
└── tests/
    ├── test_dataset_prepare.py            # Preprocessing parser & dimension tests
    ├── test_models.py                     # CNN & MobileNet forward pass tests
    ├── test_face_detector.py              # Detector & fallback path unit tests
    ├── test_inference.py                  # Inference engine tests
    └── test_pipeline_regression.py        # End-to-end training regression test with accuracy assertions
```

---

## 📚 References

- Goodfellow, I. et al. *"Challenges in Representation Learning: A report on three machine learning contests."* ICML 2013 / arXiv:1307.0414.
- Howard, A. et al. *"Searching for MobileNetV3."* ICCV 2019 / arXiv:1905.02244.
- Lugaresi, C. et al. *"MediaPipe: A Framework for Building Perception Pipelines."* arXiv:1906.08172.
