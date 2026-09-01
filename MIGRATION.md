# Migration Guide (v1.0 $\to$ v2.0)

This guide outlines breaking changes, directory restructuring, API upgrades, and migration steps for upgrading existing clones of the **Emotion Detection using Facial Recognition** repository.

---

## 🚨 Breaking Changes & Directory Restructuring

### 1. Dataset Directory Organization
- **Old Behavior**: `dataset_prepare.py` created directories under `data/train/` and `data/test/`, but attempted to save files directly to `train/` and `test/` (causing `FileNotFoundError`).
- **New Behavior**: All dataset images are consistently written to `data/train/<emotion>/` and `data/test/<emotion>/`.
- **Action Required**: If you prepared datasets with the old script, move your directories into `data/train` and `data/test` or rerun:
  ```bash
  python dataset_prepare.py --csv fer2013.csv --output data
  ```

### 2. Pinned Python & Dependencies
- **Old Dependencies**: Pinned outdated packages (`numpy==1.22.0`, `opencv-python==4.2.0.32`, `tensorflow==2.9.3`) which fail on modern Python versions (3.10+) and Apple Silicon macOS.
- **New Dependencies**: Updated to modern cross-platform pinned packages supporting Python 3.10–3.12 on Apple Silicon (`arm64`) and `x86_64`.
- **Action Required**: Recreate your virtual environment:
  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

### 3. Keras API Migration
- **Old APIs**: Deprecated `model.fit_generator()`, `Adam(lr=..., decay=...)`, and legacy `ImageDataGenerator`.
- **New APIs**: Migrated to `model.fit()`, `Adam(learning_rate=...)`, and `tf.keras.utils.image_dataset_from_directory` with `tf.keras.layers.Rescaling(1./255)`.

---

## ⚡ Key Improvements & UX Differences

1. **Normalized Live Inference**:
   - Fixed missing input scaling in live webcam loop (`[0, 255] -> [0.0, 1.0]`) via unified `preprocess_face_roi` in `src.models`.

2. **Upgraded Face Detector**:
   - Modern **MediaPipe Face Detection** (BlazeFace SSD topology) with automatic fallback to **OpenCV Haar Cascade** (`haarcascade_frontalface_default.xml`).

3. **Web UI vs CLI Webcam Modes**:
   - **Streamlit Web App (`app.py`)**: Uses **snapshot-based capture** (`st.camera_input`) for zero-dependency browser photo analysis and multi-face probability bar charts.
   - **CLI Script (`emotions.py --mode display`)**: Uses OpenCV VideoCapture for **continuous 60 FPS real-time webcam video feed**.

4. **Transfer Learning Baseline & Benchmarks**:
   - Added MobileNetV3-Small baseline (~65.8% accuracy, 23.6 ms) and `src/benchmark.py` comparing against the custom CNN (63.2% accuracy, 2.66 ms).

---

## 🚀 Quickstart Commands

```bash
# 1. Prepare FER-2013 dataset
python dataset_prepare.py --csv fer2013.csv --output data

# 2. Train CNN with data augmentation
python emotions.py --mode train --model cnn --epochs 50 --augment

# 3. Live continuous webcam inference (OpenCV 60 FPS)
python emotions.py --mode display --detector mediapipe

# 4. Launch interactive Web UI (Snapshot & Image Upload)
streamlit run app.py

# 5. Run automated test suite
pytest -v
```
