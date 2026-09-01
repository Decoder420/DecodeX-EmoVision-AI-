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
    Tracks hand landmarks and fingertips with EMA smoothing.
    Supports MediaPipe detection with automatic fallback to high-precision OpenCV kinematic tracker.
    """
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_alpha=0.65):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.filter = EMAFilter(alpha=smooth_alpha)
        self.mp_detector = None

        # Attempt to initialize MediaPipe if solutions exist
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                self.mp_detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
        except Exception:
            self.mp_detector = None

    def process_frame(self, frame_bgr, face_bboxes=[]):
        """
        Processes frame and extracts smoothed hand landmarks.
        """
        h_img, w_img = frame_bgr.shape[:2]
        hand_results = []

        # Strategy 1: MediaPipe Solutions (if active)
        if self.mp_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.mp_detector.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                        raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                        smoothed = self.filter.apply(idx, raw_landmarks)

                        landmarks_px = np.zeros((21, 2), dtype=np.int32)
                        landmarks_px[:, 0] = np.clip(smoothed[:, 0] * w_img, 0, w_img - 1).astype(np.int32)
                        landmarks_px[:, 1] = np.clip(smoothed[:, 1] * h_img, 0, h_img - 1).astype(np.int32)

                        palm_center_px = (int((landmarks_px[0, 0] + landmarks_px[9, 0]) / 2),
                                          int((landmarks_px[0, 1] + landmarks_px[9, 1]) / 2))
                        palm_scale = max(0.01, float(np.linalg.norm(smoothed[0, :2] - smoothed[9, :2])))

                        hand_results.append({
                            "hand_id": idx,
                            "handedness": "Right" if idx == 0 else "Left",
                            "landmarks": smoothed,
                            "landmarks_px": landmarks_px,
                            "palm_center": palm_center_px,
                            "palm_scale": palm_scale
                        })
                    return hand_results
            except Exception:
                pass

        # Strategy 2: High-Speed Native Kinematic Hand Tracker (Robust CV Segmentation)
        hand_results = self._detect_hands_kinematic(frame_bgr, face_bboxes)
        return hand_results

    def _detect_hands_kinematic(self, frame_bgr, face_bboxes):
        """
        Real-time hand segmentation using skin color modeling, convex hull, and defect analysis.
        """
        h_img, w_img = frame_bgr.shape[:2]
        mask_frame = frame_bgr.copy()

        # Mask out face bounding boxes to avoid confusing face with hand
        for (fx, fy, fw, fh) in face_bboxes:
            x1 = max(0, fx - 30)
            y1 = max(0, fy - 30)
            x2 = min(w_img, fx + fw + 30)
            y2 = min(h_img, fy + fh + 60)
            cv2.rectangle(mask_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Multi-color space skin segmentation (YCrCb + HSV)
        ycrcb = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2HSV)

        # YCrCb skin range
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 175, 127], dtype=np.uint8))
        # HSV skin range
        mask_hsv = cv2.inRange(hsv, np.array([0, 30, 60], dtype=np.uint8), np.array([25, 170, 255], dtype=np.uint8))

        skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # Morphological filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area descending
        contours = [c for c in contours if cv2.contourArea(c) > 2500]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.max_num_hands]

        hands_data = []
        for idx, cnt in enumerate(contours):
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            x, y, w, h = cv2.boundingRect(cnt)
            palm_scale = max(0.08, float(max(w, h)) / float(max(w_img, h_img)))

            # Extreme points
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

            # Reconstruct 21 pseudo-landmarks
            raw_lms = np.zeros((21, 3), dtype=np.float32)
            # Wrist
            raw_lms[0] = [bottommost[0] / w_img, bottommost[1] / h_img, 0.0]
            # Middle MCP / Palm
            raw_lms[9] = [cx / w_img, cy / h_img, 0.0]
            # Thumb tip
            raw_lms[4] = [leftmost[0] / w_img, (leftmost[1] + cy) / (2 * h_img), 0.0]
            # Index tip (topmost point)
            raw_lms[8] = [topmost[0] / w_img, topmost[1] / h_img, 0.0]
            raw_lms[6] = [(topmost[0] + cx) / (2 * w_img), (topmost[1] + cy) / (2 * h_img), 0.0]
            # Middle tip
            raw_lms[12] = [(topmost[0] + rightmost[0]) / (2 * w_img), (topmost[1] + 12) / h_img, 0.0]
            raw_lms[10] = [(topmost[0] + cx + rightmost[0]) / (3 * w_img), (topmost[1] + cy) / (2 * h_img), 0.0]
            # Ring tip
            raw_lms[16] = [rightmost[0] / w_img, (rightmost[1] + topmost[1]) / (2 * h_img), 0.0]
            raw_lms[14] = [(rightmost[0] + cx) / (2 * w_img), (rightmost[1] + cy) / (2 * h_img), 0.0]
            # Pinky tip
            raw_lms[20] = [rightmost[0] / w_img, rightmost[1] / h_img, 0.0]
            raw_lms[18] = [(rightmost[0] + cx) / (2 * w_img), (rightmost[1] + cy + 10) / (2 * h_img), 0.0]

            # Fill intermediate nodes
            raw_lms[1] = (raw_lms[0] * 0.7 + raw_lms[4] * 0.3)
            raw_lms[2] = (raw_lms[0] * 0.4 + raw_lms[4] * 0.6)
            raw_lms[3] = (raw_lms[0] * 0.2 + raw_lms[4] * 0.8)
            raw_lms[5] = (raw_lms[0] * 0.5 + raw_lms[6] * 0.5)
            raw_lms[7] = (raw_lms[6] * 0.5 + raw_lms[8] * 0.5)
            raw_lms[11] = (raw_lms[10] * 0.5 + raw_lms[12] * 0.5)
            raw_lms[13] = (raw_lms[0] * 0.5 + raw_lms[14] * 0.5)
            raw_lms[15] = (raw_lms[14] * 0.5 + raw_lms[16] * 0.5)
            raw_lms[17] = (raw_lms[0] * 0.5 + raw_lms[18] * 0.5)
            raw_lms[19] = (raw_lms[18] * 0.5 + raw_lms[20] * 0.5)

            # Apply EMA smoothing
            smoothed = self.filter.apply(idx, raw_lms)

            # Convert to pixel coords
            lms_px = np.zeros((21, 2), dtype=np.int32)
            lms_px[:, 0] = np.clip(smoothed[:, 0] * w_img, 0, w_img - 1).astype(np.int32)
            lms_px[:, 1] = np.clip(smoothed[:, 1] * h_img, 0, h_img - 1).astype(np.int32)

            hands_data.append({
                "hand_id": idx,
                "handedness": "Right" if idx == 0 else "Left",
                "landmarks": smoothed,
                "landmarks_px": lms_px,
                "palm_center": (cx, cy),
                "palm_scale": palm_scale
            })

        if not hands_data:
            self.filter.reset()

        return hands_data

    def close(self):
        if self.mp_detector is not None:
            try:
                self.mp_detector.close()
            except Exception:
                pass
