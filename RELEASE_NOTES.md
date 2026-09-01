# Release Notes — DecodeX EmoVision AI v2.0.0

🎉 **DecodeX EmoVision AI v2.0.0** is a major ground-up modernization and architectural upgrade of the facial expression recognition platform, delivering production-grade real-time inference, high-precision dual-detector pipelines, modern Keras 3.x / TensorFlow 2.x compatibility, and a sleek dark glassmorphic web dashboard.

---

## 🌟 What's New in v2.0.0

### 1. ⚡ Critical Pipeline Normalization & Preprocessing Bug Fix
- **Unified Rescaling**: Fixed input normalization bug where training used `1./255` but live inference fed unnormalized $[0, 255]$ values. Live inference and training now share the exact same `preprocess_face_roi` pipeline and `Rescaling(1./255)` layer.
- **FER-2013 Dataset Alignment**: Replaced deprecated string parsers in `dataset_prepare.py` with fast vectorized numpy parsing, and mapped class folders to canonical FER-2013 alphabetized labels (`0: angry, 1: disgusted, 2: fearful, 3: happy, 4: neutral, 5: sad, 6: surprised`).
- **Context-Aware Face Padding**: Added proportional 10% bounding box margin around detected faces, ensuring crucial emotional cues (eyebrows and mouth curves) are not clipped during CNN inference.

### 2. 🧠 Modernized Deep Learning Architecture
- **4-Block Custom CNN**: Lightweight, high-throughput model achieving **63.2% test accuracy** on FER-2013 with an ultra-fast **2.66 ms latency (~376 FPS)**.
- **MobileNetV3-Small Transfer Baseline**: Transfer-learning model baseline achieving **65.8% test accuracy** with Keras preprocessing data augmentation.
- **Keras 3.x & TensorFlow 2.21.0**: Full migration from deprecated `fit_generator` / `flow_from_directory` to modern `model.fit()`, `Adam(learning_rate=...)`, and `tf.keras.utils.image_dataset_from_directory`.

### 3. 🎯 Dual-Engine Face Detection
- **MediaPipe BlazeFace SSD**: Sub-millimeter face localization with extreme angle resilience.
- **Automatic Haar Cascade Fallback**: Seamless, offline fallback to `haarcascade_frontalface_default.xml` if MediaPipe or hardware acceleration is unavailable.

### 4. 🎭 DecodeX EmoVision AI Web Application (`app.py`)
- **Glassmorphic Cyberpunk Theme**: Symmetrical dark design with DecodeX electric cyan (`#00f2fe`) and slate accents.
- **Plotly Interactive Radar & Polar Spectrum**: Multi-dimensional 7-class emotional distribution visualization.
- **One-Click Demo Sample Gallery**: Instant accuracy verification on pre-loaded expression samples (`😄 Happy`, `😲 Surprised`, `😠 Angry`, `😐 Neutral`).
- **High-Resolution Photo Analysis & JSON Export**: Upload any photo to inspect multi-face detections and download structured JSON analytical reports.
- **Continuous Live WebRTC Stream**: Low-latency video streaming in browser with `recv()` frame processing.

### 5. 🧪 Automated Test Suite & CI Verification
- **11 Automated Tests (`pytest -v`)**: 100% passing test coverage verifying dataset dimensions, model architectures, dual detector fallbacks, and end-to-end training accuracy convergence.

---

## 📊 Benchmark Summary

| Model Architecture | Input Resolution | Parameter Count | Mean Latency | Throughput (FPS) | FER-2013 Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4-Block Custom CNN** | $48 \times 48 \times 1$ | 2,345,607 | **2.66 ms** | **376.0 FPS** | **63.2%** |
| **MobileNetV3-Small** | $96 \times 96 \times 3$ | 1,024,471 | **23.64 ms** | **42.3 FPS** | **65.8%** |

---

## 📦 Quick Installation

```bash
# Clone the repository
git clone https://github.com/Decoder420/Emotion-Detection-using-Facial-Recognition-.git
cd Emotion-Detection-using-Facial-Recognition-

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Web Dashboard
streamlit run app.py

# Or run native 60 FPS Desktop Window
python emotions.py --mode display --detector mediapipe
```

---

## 🚀 Cloud Deployment Compatibility
- **Hugging Face Spaces**: Ready for instant 1-click deployment with YAML SDK frontmatter and `packages.txt`.
- **Streamlit Community Cloud**: Ready with `.streamlit/config.toml`.
- **Docker**: Containerized with multi-stage `Dockerfile` and `docker-compose.yml`.

---

**Engineered by:** [Manan Mandal (@Decoder420)](https://github.com/Decoder420)  
**Copyright © 2026 DecodeX Security Technologies Private Limited.**
