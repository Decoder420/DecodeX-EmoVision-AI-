import cv2
import numpy as np
from src.inference import EmotionEngine, EMOTION_COLORS
from src.models import EMOTION_DICT

def test_emotion_engine_process_frame():
    # Initialize engine (without saved weights, runs forward pass on uninitialized model)
    engine = EmotionEngine(model_path="nonexistent_model.h5", model_type="cnn", detector_type="haar")

    # Synthetic image with a bright rectangle to simulate ROI
    frame = np.full((300, 300, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (80, 80), (220, 220), (200, 200, 200), -1)

    annotated, results = engine.process_frame(frame, draw_annotations=True)

    assert annotated.shape == frame.shape
    assert isinstance(results, list)

    # If a face or box was found, assert result structure
    for r in results:
        assert "bbox" in r
        assert "emotion" in r
        assert r["emotion"] in EMOTION_DICT.values()
        assert "confidence" in r
        assert 0.0 <= r["confidence"] <= 1.0
        assert "probabilities" in r
        assert len(r["probabilities"]) == 7
