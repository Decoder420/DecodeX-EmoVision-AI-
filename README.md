---
title: DecodeX EmoVision AI - Facial Emotion Intelligence
emoji: 🎭
colorFrom: cyan
colorTo: blue
sdk: streamlit
sdk_version: "1.62.0"
app_file: app.py
pinned: false
license: mit
---

<p align="center">
  <img src="assets/logo/decodex_banner.png" alt="DecodeX Logo" width="460" />
</p>

<h1 align="center">DecodeX EmoVision AI</h1>
<p align="center"><strong>Next-Generation Facial Emotion Perception & Expression Intelligence</strong></p>

<p align="center">
  <a href="https://decodex-emovision-ai.onrender.com/"><img src="https://img.shields.io/badge/🚀_Live_Demo-Render_Cloud-00f2fe?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo on Render" /></a>
  <a href="https://github.com/Decoder420/DecodeX-EmoVision-AI-"><img src="https://img.shields.io/badge/GitHub-DecodeX--EmoVision--AI--181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repository" /></a>
  <a href="https://github.com/Decoder420"><img src="https://img.shields.io/badge/Engineered_By-Decoder420-6366f1?style=for-the-badge&logo=github" alt="Engineer" /></a>
</p>

> 🌐 **Live Web Application**: **[https://decodex-emovision-ai.onrender.com/](https://decodex-emovision-ai.onrender.com/)**

A deep learning facial expression recognition system capable of classifying human facial expressions into **seven emotions** (`Angry`, `Disgusted`, `Fearful`, `Happy`, `Neutral`, `Sad`, `Surprised`) in real time from live webcam feeds, uploaded photos, or the **FER-2013** dataset.

![Accuracy Plot](accuracy.png)

---

## 🌟 Key Features

- **🌐 Cloud-Deployed Live Web App**: Accessible worldwide with full webcam permissions at [https://decodex-emovision-ai.onrender.com/](https://decodex-emovision-ai.onrender.com/).
- **📹 Continuous 60 FPS Live Stream**: Integrated real-time Face-API neural network with dynamic cyber-HUD tracking, bounding brackets, and live emotion confidence meters.
- **📸 High-Precision Neural Snapshot**: Single-frame deep scan with full 7-class Plotly polar radar spectrum and horizontal distribution charts.
- **✨ One-Click Sample Gallery**: Pre-loaded expression presets (Happy 😄, Surprised 😲, Angry 😠, Neutral 😐) for instant testing.
- **🖼️ High-Res Image Upload & JSON Export**: Multi-face detection, batch analysis, and one-click JSON telemetry report export.
- **Robust Face Detection**: High-accuracy **MediaPipe Face Detection** (BlazeFace SSD topology) with automatic offline fallback to classical **OpenCV Haar Cascades** (`haarcascade_frontalface_default.xml`).
- **Modernized Deep Learning Stack**: Full compatibility with **TensorFlow 2.x / Keras 3.x**, Python 3.10–3.12, and Apple Silicon (`arm64`) as well as `x86_64`.
- **Architectures**:
  - **4-Block Custom CNN**: Lightweight model (~63.2% test accuracy on FER-2013, 2.66 ms latency / 376 FPS).
  - **MobileNetV3-Small Baseline**: Transfer-learning baseline (~65.8% test accuracy, 23.64 ms latency / 42 FPS) with data augmentation layers.
- **Automated Test Suite**: Full `pytest` verification covering dataset preprocessing, model architectures, face detection fallbacks, and end-to-end training regression.

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Decoder420/DecodeX-EmoVision-AI-.git
   cd DecodeX-EmoVision-AI-
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

### 1. Interactive Web Application (Streamlit)
Launch the Streamlit web dashboard locally:
```bash
streamlit run app.py
```
Or view the live cloud version: **[https://decodex-emovision-ai.onrender.com/](https://decodex-emovision-ai.onrender.com/)**

### 2. Continuous Live Webcam Feed (OpenCV GUI)
Real-time 60 FPS continuous video inference in a dedicated desktop OpenCV window:
```bash
python emotions.py --mode display --detector haar
```

### 3. Data Preparation (FER-2013)
Download `fer2013.csv` from [Kaggle](https://www.kaggle.com/deadskull7/fer2013) into the project root, then run:
```bash
python dataset_prepare.py --csv ./fer2013.csv --output data
```

### 4. Model Training
Train the CNN or MobileNetV3 model with data augmentation and modern Keras callbacks:
```bash
# Train standard 4-block CNN
python emotions.py --mode train --model cnn --epochs 50 --augment

# Train MobileNetV3-Small
python emotions.py --mode train --model mobilenet --epochs 50 --augment
```

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

---

## 📁 Repository Structure

```
DecodeX-EmoVision-AI-/
├── app.py                                 # Streamlit web application (5-tab HUD)
├── dataset_prepare.py                     # FER-2013 CSV parser & image generator
├── emotions.py                            # Main CLI training & webcam display script
├── haarcascade_frontalface_default.xml    # Permanent offline Haar Cascade fallback asset
├── model.h5                               # Pre-trained 4-block CNN weights (9.0 MB)
├── render.yaml                            # 1-click cloud deployment blueprint for Render
├── requirements.txt                       # Pinned dependencies for Python 3.10-3.12
├── RELEASE_NOTES.md                       # Comprehensive release notes (v2.0.0)
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
    └── test_pipeline_regression.py        # End-to-end training regression test
```

---

## 🛡️ License & Credits

- **Engineered by**: [**Manan Mandal (@Decoder420)**](https://github.com/Decoder420)
- **Organization**: **DecodeX Security Technologies Private Limited**
- **License**: [MIT License](LICENSE)
