import cv2
import numpy as np
import os

class FaceDetector:
    """
    Face Detector supporting MediaPipe and OpenCV Haar Cascade fallback.
    """
    def __init__(self, method="mediapipe", min_detection_confidence=0.5, cascade_path="haarcascade_frontalface_default.xml"):
        self.method = method.lower()
        self.min_detection_confidence = min_detection_confidence
        self.cascade_path = cascade_path
        self.detector = None
        self.haar_cascade = None

        if self.method == "mediapipe":
            try:
                import mediapipe as mp
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                    self.mp_face_detection = mp.solutions.face_detection
                    self.detector = self.mp_face_detection.FaceDetection(
                        model_selection=0,
                        min_detection_confidence=self.min_detection_confidence
                    )
                else:
                    self.method = "haar"
            except Exception as e:
                print(f"[INFO] Using Haar Cascade detector ({e})")
                self.method = "haar"

        if self.method == "haar" or self.detector is None:
            self.method = "haar"
            search_paths = [
                self.cascade_path,
                os.path.join(os.path.dirname(__file__), "..", self.cascade_path),
                os.path.join(os.path.dirname(__file__), self.cascade_path)
            ]
            for p in search_paths:
                if os.path.exists(p):
                    self.cascade_path = p
                    break
            self.haar_cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, frame):
        """
        Detects faces in a BGR video frame or image.
        Returns a list of bounding boxes [(x, y, w, h), ...] in pixel coordinates.
        """
        h_img, w_img = frame.shape[:2]
        faces = []

        if self.method == "mediapipe" and self.detector is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w_img))
                    y = max(0, int(bbox.ymin * h_img))
                    w = min(w_img - x, int(bbox.width * w_img))
                    h = min(h_img - y, int(bbox.height * h_img))
                    if w > 10 and h > 10:
                        faces.append((x, y, w, h))
        
        # If mediapipe returned no faces or haar was selected
        if not faces and self.haar_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # High-sensitivity detection parameters
            detected = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(30, 30)
            )
            for (x, y, w, h) in detected:
                faces.append((int(x), int(y), int(w), int(h)))

        return faces

    def close(self):
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
