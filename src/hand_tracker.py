import cv2
import numpy as np
import os

class EMAFilter:
    """
    Exponential Moving Average (EMA) filter for smoothing landmark coordinates.
    Eliminates frame-to-frame jitter while maintaining high responsiveness.
    """
    def __init__(self, alpha=0.65):
        self.alpha = alpha
        self.smoothed_landmarks = {}

    def apply(self, hand_id, landmarks):
        """
        Applies EMA smoothing to an array or list of (x, y, z) landmarks.
        landmarks: np.ndarray of shape (21, 3)
        """
        if hand_id not in self.smoothed_landmarks:
            self.smoothed_landmarks[hand_id] = np.array(landmarks, dtype=np.float32)
            return self.smoothed_landmarks[hand_id]

        prev = self.smoothed_landmarks[hand_id]
        curr = np.array(landmarks, dtype=np.float32)
        smoothed = self.alpha * curr + (1.0 - self.alpha) * prev
        self.smoothed_landmarks[hand_id] = smoothed
        return smoothed

    def reset(self, hand_id=None):
        if hand_id is not None:
            self.smoothed_landmarks.pop(hand_id, None)
        else:
            self.smoothed_landmarks.clear()

class HandTracker:
    """
    Tracks 21 hand landmarks per hand with EMA coordinate smoothing.
    """
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_alpha=0.65):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.filter = EMAFilter(alpha=smooth_alpha)
        self.mp_hands = None
        self.detector = None

        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                self.mp_hands = mp.solutions.hands
                self.detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
        except Exception as e:
            print(f"[INFO] MediaPipe Hands initialized with fallback mode: {e}")

    def process_frame(self, frame_bgr):
        """
        Processes a BGR image/frame and extracts smoothed landmarks for detected hands.
        Returns:
            List of dicts: [
                {
                    "hand_id": int,
                    "handedness": "Left" | "Right",
                    "landmarks": np.ndarray (21, 3) in normalized [0, 1] coords,
                    "landmarks_px": np.ndarray (21, 2) in pixel (x, y) coords,
                    "palm_center": (x_px, y_px),
                    "palm_scale": float
                }, ...
            ]
        """
        h_img, w_img = frame_bgr.shape[:2]
        hand_results = []

        if self.detector is None:
            return hand_results

        # Hand tracking is optimized on downscaled image for performance
        downscale_w = min(w_img, 480)
        scale_ratio = downscale_w / float(w_img)
        downscale_h = int(h_img * scale_ratio)
        resized_frame = cv2.resize(frame_bgr, (downscale_w, downscale_h), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                raw_landmarks = []
                for lm in hand_lms.landmark:
                    raw_landmarks.append([lm.x, lm.y, lm.z])
                raw_landmarks = np.array(raw_landmarks, dtype=np.float32)

                # Apply EMA Smoothing
                smoothed = self.filter.apply(idx, raw_landmarks)

                # Convert to pixel coordinates on original frame size
                landmarks_px = np.zeros((21, 2), dtype=np.int32)
                landmarks_px[:, 0] = np.clip(smoothed[:, 0] * w_img, 0, w_img - 1).astype(np.int32)
                landmarks_px[:, 1] = np.clip(smoothed[:, 1] * h_img, 0, h_img - 1).astype(np.int32)

                # Handedness label
                handedness = "Right"
                if results.multi_handedness and len(results.multi_handedness) > idx:
                    handedness = results.multi_handedness[idx].classification[0].label

                # Compute palm center (midpoint between Wrist #0 and Middle MCP #9)
                palm_center_px = (
                    int((landmarks_px[0, 0] + landmarks_px[9, 0]) / 2),
                    int((landmarks_px[0, 1] + landmarks_px[9, 1]) / 2)
                )

                # Palm scale: distance between wrist (#0) and middle MCP (#9)
                palm_scale = float(np.linalg.norm(smoothed[0, :2] - smoothed[9, :2]))
                palm_scale = max(0.01, palm_scale)

                hand_results.append({
                    "hand_id": idx,
                    "handedness": handedness,
                    "landmarks": smoothed,
                    "landmarks_px": landmarks_px,
                    "palm_center": palm_center_px,
                    "palm_scale": palm_scale
                })
        else:
            self.filter.reset()

        return hand_results

    def close(self):
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
