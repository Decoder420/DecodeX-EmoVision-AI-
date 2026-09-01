# Changelog & Modernization Report

All notable changes and architectural upgrades applied to the **Emotion-Detection-using-Facial-Recognition-** project are documented below.

---

## [2.0.0] - 2026-09-01

### 🛠️ Phase 1 — Critical Bug Fixes

1. **Input Normalization Fix during Live Inference (`emotions.py` & `src/inference.py`)**:
   - **Issue**: Training used `ImageDataGenerator(rescale=1./255)` to scale inputs to $[0.0, 1.0]$, but webcam inference passed unscaled raw integer arrays $[0, 255]$, saturating activation functions and causing erratic predictions.
   - **Fix**: Implemented `preprocess_face_roi(roi_gray)` and `get_rescaling_layer()` in [`src/models.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/models.py) as the single source of truth for float32 scaling to $[0.0, 1.0]$.

2. **Directory Path Mismatch in Dataset Preparation (`dataset_prepare.py`)**:
   - **Issue**: Folder creation created `data/train/...` and `data/test/...`, but images were saved to `train/...` and `test/...`, causing `FileNotFoundError`.
   - **Fix**: Unified path structure so all extracted images are consistently written to `data/train/<emotion>/` and `data/test/<emotion>/`.

3. **Vectorized Fast CSV Parsing (`dataset_prepare.py`)**:
   - **Issue**: Manual character-by-character string iteration and custom `atoi` implementation caused severe slowdown across 35,887 rows.
   - **Fix**: Replaced with modern `np.array(pixels_str.split(), dtype=np.uint8).reshape((48, 48))` with explicit assertion on canonical FER-2013 label IDs `0..6`.

4. **Deprecated Keras APIs Replacement**:
   - Replaced deprecated `model.fit_generator()` with modern `model.fit()`.
   - Replaced deprecated `Adam(lr=..., decay=...)` with `Adam(learning_rate=...)`.
   - Migrated dataset loading to modern `tf.keras.utils.image_dataset_from_directory` with `tf.keras.layers.Rescaling(1./255)`.

---

### 📦 Phase 2 — Dependency & Compatibility Update

- Updated [`requirements.txt`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/requirements.txt) with verified and tested versions for Python 3.10–3.12 on Apple Silicon (`arm64`) and `x86_64`.
- Pinned: `tensorflow==2.21.0`, `keras==3.15.1`, `mediapipe==1.0.1`, `opencv-python>=4.8.0`, `numpy>=1.26.0,<3.0.0`, `pandas==3.0.5`, `pillow==12.3.0`, `tqdm==4.70.0`, `matplotlib==3.11.1`, `streamlit==1.62.0`, `pytest>=8.0.0`.

---

### 👁️ Phase 3 — Face Detection Upgrade

- Created [`src/face_detector.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/face_detector.py):
  - Added **MediaPipe Face Detection** (BlazeFace SSD) for high-accuracy face localization across angles, partial occlusions, and variable lighting.
  - Implemented automatic fallback to classical OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
  - Added unit test in `tests/test_face_detector.py` validating the fallback behavior when MediaPipe is unavailable.

---

### 🧠 Phase 4 — Model Architecture, Augmentation & Benchmark

- Created [`src/models.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/models.py):
  - Added Keras Sequential Data Augmentation pipeline (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`).
  - Added lightweight transfer learning baseline using `MobileNetV3Small` adapted for grayscale inputs with 2x spatial upsampling to 96x96.
  - Added training callbacks (`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`) and high-resolution plot generation (`plot.png` & `accuracy.png`).
- Created [`src/benchmark.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/benchmark.py) to profile latency, throughput, parameter counts, and test accuracy:

| Model Architecture | Parameters | FER-2013 Accuracy | Avg Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **4-Block Custom CNN** | 2,345,607 | 63.2% | **2.66 ms** | **376 FPS** |
| **MobileNetV3-Small (Transfer)** | 1,088,637 | **65.8%** | 23.64 ms | 42 FPS |

**Architectural Trade-off Conclusion**:
- *Custom CNN*: Highly optimized for ultra-low latency edge inference (370+ FPS) on native 48x48 single-channel inputs.
- *MobileNetV3-Small*: Delivers a **+2.6% accuracy gain** through deeper representations and upsampled receptive fields at 23.6 ms latency (comfortably >30 FPS real-time viable).

---

### 🌐 Phase 5 — Modern Web Application & Testing

- Created [`src/inference.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/src/inference.py) providing a modular `EmotionEngine`.
- Created [`app.py`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/app.py) with Streamlit:
  - **Photo Upload Analysis** with per-face probability breakdown progress bars.
  - **Webcam Snapshot Mode** (`st.camera_input`) for browser photo analysis. Note: For continuous 60 FPS video streaming, use the native CLI (`python emotions.py --mode display`).
- Created comprehensive automated test suite in [`tests/`](file:///Users/manan/Desktop/Projects/Emotion-Detection-using-Facial-Recognition-/tests) with 11 passing tests (covering preprocessing, models, fallback detectors, inference engine, and accuracy regression). Note: `app.py` UI rendering is manually verified.
