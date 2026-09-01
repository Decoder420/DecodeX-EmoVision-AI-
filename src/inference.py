import cv2
import numpy as np
import os
import tensorflow as tf
from src.models import build_cnn_model, build_mobilenet_model, preprocess_face_roi, EMOTION_DICT
from src.face_detector import FaceDetector

EMOTION_COLORS = {
    "Angry": (0, 0, 220),       # Red
    "Disgusted": (0, 140, 70),   # Dark Green
    "Fearful": (180, 0, 180),    # Purple
    "Happy": (0, 215, 255),      # Gold / Yellow
    "Neutral": (160, 160, 160),  # Gray
    "Sad": (255, 140, 0),        # Deep Blue
    "Surprised": (255, 105, 180) # Pink
}

class EmotionEngine:
    """
    Unified Inference Engine for Face Emotion Classification.
    Reused across CLI and Web UI pipelines.
    """
    def __init__(self, model_path="model.h5", model_type="cnn", detector_type="mediapipe", cascade_path="haarcascade_frontalface_default.xml"):
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.detector = FaceDetector(method=detector_type, cascade_path=cascade_path)
        self.model = self._load_model()

    def _load_model(self):
        """
        Builds architecture and loads weights if available.
        """
        if self.model_type == "mobilenet":
            model = build_mobilenet_model(input_shape=(48, 48, 1), num_classes=7)
        else:
            model = build_cnn_model(input_shape=(48, 48, 1), num_classes=7)

        # Resolve model path if in parent or relative dir
        search_paths = [
            self.model_path,
            os.path.join(os.path.dirname(__file__), "..", self.model_path),
            os.path.join(os.path.dirname(__file__), self.model_path)
        ]
        found_path = None
        for p in search_paths:
            if os.path.exists(p):
                found_path = p
                break

        if found_path is not None:
            try:
                model.load_weights(found_path)
                print(f"[INFO] Loaded model weights from {found_path}")
            except Exception as e:
                print(f"[WARN] Error loading weights directly ({e}), attempting load_model...")
                try:
                    model = tf.keras.models.load_model(found_path)
                except Exception as e2:
                    print(f"[ERROR] Failed to load model weights: {e2}")
        else:
            print(f"[WARN] Model weights file not found at '{self.model_path}'. Running with uninitialized weights.")

        return model

    def process_frame(self, frame_bgr, draw_annotations=True):
        """
        Processes a single BGR frame/image:
        1. Detects faces
        2. Crops & normalizes to (48, 48) using unified preprocess_face_roi with face padding
        3. Predicts emotion probabilities
        4. Draws annotations if requested

        Returns:
            annotated_frame: Frame with drawn bounding boxes and labels
            results: List of dicts containing bbox, emotion, confidence, and full probabilities
        """
        annotated_frame = frame_bgr.copy() if draw_annotations else frame_bgr
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect_faces(frame_bgr)
        results = []

        h_img, w_img = frame_bgr.shape[:2]

        for (x, y, w, h) in faces:
            # Add 10% proportional padding to capture the full facial expression
            pad_x = int(0.10 * w)
            pad_y = int(0.10 * h)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            roi_gray = gray_frame[y1:y2, x1:x2]
            if roi_gray.size == 0 or roi_gray.shape[0] < 5 or roi_gray.shape[1] < 5:
                continue

            # SINGLE SOURCE OF TRUTH: preprocess_face_roi handles resizing, dim expansion, and float32 scaling to [0.0, 1.0]
            preprocessed_roi = preprocess_face_roi(roi_gray)

            # Predict emotion
            predictions = self.model.predict(preprocessed_roi, verbose=0)[0]
            max_idx = int(np.argmax(predictions))
            confidence = float(predictions[max_idx])
            emotion_label = EMOTION_DICT.get(max_idx, "Unknown")

            prob_dict = {EMOTION_DICT[i]: float(prob) for i, prob in enumerate(predictions)}

            results.append({
                "bbox": (x, y, w, h),
                "emotion": emotion_label,
                "confidence": confidence,
                "probabilities": prob_dict
            })

            if draw_annotations:
                color = EMOTION_COLORS.get(emotion_label, (255, 255, 255))
                # Bounding box around detected face
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                
                # Emotion label and confidence badge
                label_text = f"{emotion_label}: {confidence*100:.1f}%"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(annotated_frame, (x, max(0, y - th - 10)), (x + tw + 12, y), color, -1)
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x + 6, max(th + 2, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        return annotated_frame, results
