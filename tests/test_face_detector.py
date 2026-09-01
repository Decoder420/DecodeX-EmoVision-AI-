import os
import cv2
import numpy as np
from unittest.mock import patch
from src.face_detector import FaceDetector

def create_synthetic_face_image():
    # Create simple 400x400 image with a basic face-like oval and eyes/mouth
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    # Head
    cv2.ellipse(img, (200, 200), (80, 110), 0, 0, 360, (150, 150, 150), -1)
    # Eyes
    cv2.circle(img, (170, 170), 10, (50, 50, 50), -1)
    cv2.circle(img, (230, 170), 10, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (200, 240), (30, 15), 0, 0, 180, (50, 50, 50), 3)
    return img

def test_mediapipe_detector_initialization():
    detector = FaceDetector(method="mediapipe")
    assert detector.method in ["mediapipe", "haar"]
    detector.close()

def test_haar_cascade_detector_fallback():
    # Explicitly test Haar cascade detector
    detector = FaceDetector(method="haar", cascade_path="haarcascade_frontalface_default.xml")
    assert detector.method == "haar"
    assert detector.haar_cascade is not None
    assert not detector.haar_cascade.empty()

    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detector.detect_faces(test_img)
    assert isinstance(faces, list)

def test_forced_fallback_when_mediapipe_fails():
    """
    Unit test verifying that when MediaPipe fails or is unavailable,
    FaceDetector automatically and gracefully falls back to Haar Cascade.
    """
    with patch.dict('sys.modules', {'mediapipe': None}):
        # Mocking mediapipe import failure
        detector = FaceDetector(method="mediapipe", cascade_path="haarcascade_frontalface_default.xml")
        assert detector.method == "haar", "Failed to fall back to Haar cascade when mediapipe failed"
        assert detector.haar_cascade is not None
        assert not detector.haar_cascade.empty()
        
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        faces = detector.detect_faces(test_img)
        assert isinstance(faces, list)
