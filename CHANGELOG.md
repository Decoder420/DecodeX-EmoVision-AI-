# Changelog & Modernization Report

All notable changes and architectural upgrades applied to the **Emotion-Detection-using-Facial-Recognition-** project are documented below.

---

## [2.0.0] - 2026-09-01

### 🛠️ Phase 1 — Critical Bug Fixes

1. **Input Normalization Fix during Live Inference (`emotions.py` & `src/inference.py`)**:
   - **Issue**: Training used `ImageDataGenerator(rescale=1./255)` to scale inputs to $[0.0, 1.0]$, but webcam inference passed unscaled raw integer arrays $[0, 255]$, saturating activation functions and causing erratic predictions.
   - **Fix**: Added explicit float32 normalization `cropped_img = cropped_img.astype("float32") / 255.0` prior to `model.predict(...)`.

2. **Directory Path Mismatch in Dataset Preparation (`dataset_prepare.py`)**:
   - **Issue**: Folder creation created `data/train/...` and `data/test/...`, but images were saved to `train/...` and `test/...`, causing `FileNotFoundError`.
   - **Fix**: Unified path structure so all extracted images are consistently written to `data/train/<emotion>/` and `data/test/<emotion>/`.

3. **Vectorized Fast CSV Parsing (`dataset_prepare.py`)**:
   - **Issue**: Manual character-by-character string iteration and custom `atoi` implementation caused severe slowdown across 35,887 rows.
   - **Fix**: Replaced with vectorized `np.fromstring(pixels_str, dtype=np.uint8, sep=' ').reshape((48, 48))`.

4. **Deprecated Keras APIs Replacement**:
   - Replaced deprecated `model.fit_generator()` with modern `model.fit()`.
   - Replaced deprecated `Adam(lr=..., decay=...)` with `Adam(learning_rate=...)`.
   - Migrated dataset loading to modern `tf.keras.utils.image_dataset_from_directory` with `tf.keras.layers.Rescaling(1./255)`.

---

### 📦 Phase 2 — Dependency & Compatibility Update

- Updated `requirements.txt` to support modern Python (>=3.10) on Apple Silicon (`arm64`) and `x86_64` platforms.
- Verified support for TensorFlow 2.x, OpenCV 4.x, NumPy, Pandas, Pillow, Matplotlib, MediaPipe, and Streamlit.

---

### 👁️ Phase 3 — Face Detection Upgrade

- Created [`src/face_detector.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/face_detector.py):
  - Added **MediaPipe Face Detection** (BlazeFace SSD) for high-accuracy face localization across angles, partial occlusions, and variable lighting.
  - Implemented automatic fallback to classical OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) if MediaPipe is unavailable.

---

### 🧠 Phase 4 — Model & Training Improvements

- Created [`src/models.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/models.py):
  - Added Keras Sequential Data Augmentation pipeline (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`) to reduce overfitting.
  - Added lightweight transfer learning baseline using `MobileNetV3Small` adapted for $48 \times 48$ grayscale inputs.
  - Added training callbacks (`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`) and high-resolution plot generation (`plot.png` & `accuracy.png`).
- Created [`src/benchmark.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/benchmark.py) to measure FLOPs, parameter count, and inference latency (ms / FPS).

---

### 🌐 Phase 5 — Modern Web Application & Modularization

- Created [`src/inference.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/inference.py):
  - Modular `EmotionEngine` that encapsulates face detection, ROI preprocessing, normalization, prediction, and visual annotation.
- Created [`app.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/app.py):
  - Sleek, dark-themed Streamlit web interface.
  - Supports **Photo Upload Analysis** with per-face probability breakdown progress bars.
  - Supports **Webcam Snapshot Mode** for instant facial expression recognition.
  - Live switcher for CNN vs MobileNetV3 architectures and MediaPipe vs Haar Cascade backends.
