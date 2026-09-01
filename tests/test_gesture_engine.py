import time
import numpy as np
import pytest
from src.hand_tracker import EMAFilter
from src.gesture_engine import GestureEngine

def test_ema_filter_smoothing():
    filter_inst = EMAFilter(alpha=0.5)

    # Initial state
    lm1 = np.zeros((21, 3), dtype=np.float32)
    s1 = filter_inst.apply(0, lm1)
    np.testing.assert_allclose(s1, lm1)

    # Second step with step input = 1.0
    lm2 = np.ones((21, 3), dtype=np.float32)
    s2 = filter_inst.apply(0, lm2)

    # Expected: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    np.testing.assert_allclose(s2, np.full((21, 3), 0.5, dtype=np.float32))

def test_pinch_gesture_trigger():
    engine = GestureEngine()

    # Synthetic hand where thumb tip #4 and index tip #8 are almost touching (pinch)
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = [0.5, 0.8, 0.0]   # Wrist
    lms[9] = [0.5, 0.5, 0.0]   # Middle MCP
    lms[4] = [0.50, 0.40, 0.0] # Thumb tip
    lms[8] = [0.51, 0.40, 0.0] # Index tip (distance = 0.01)

    hand_data = [{
        "hand_id": 0,
        "handedness": "Right",
        "landmarks": lms,
        "landmarks_px": np.zeros((21, 2), dtype=np.int32),
        "palm_center": (320, 240),
        "palm_scale": 0.3
    }]

    result = engine.analyze(hand_data)
    assert result["pinch_active"] is True
    assert "PINCH_CLICK" in result["gestures"]

def test_open_palm_gesture():
    engine = GestureEngine()

    # Synthetic hand where all fingers are fully extended upwards
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = [0.5, 0.9, 0.0] # Wrist

    # Tips extended high above PIPs
    lms[4] = [0.3, 0.6, 0.0]  # Thumb tip
    lms[8] = [0.4, 0.3, 0.0]  # Index tip
    lms[6] = [0.4, 0.6, 0.0]  # Index PIP
    lms[12] = [0.5, 0.25, 0.0] # Middle tip
    lms[10] = [0.5, 0.6, 0.0] # Middle PIP
    lms[16] = [0.6, 0.3, 0.0]  # Ring tip
    lms[14] = [0.6, 0.6, 0.0]  # Ring PIP
    lms[20] = [0.7, 0.35, 0.0] # Pinky tip
    lms[18] = [0.7, 0.65, 0.0] # Pinky PIP
    lms[9] = [0.5, 0.6, 0.0]  # Middle MCP

    hand_data = [{
        "hand_id": 0,
        "handedness": "Right",
        "landmarks": lms,
        "landmarks_px": np.zeros((21, 2), dtype=np.int32),
        "palm_center": (320, 240),
        "palm_scale": 0.3
    }]

    result = engine.analyze(hand_data)
    assert "OPEN_PALM" in result["gestures"]

def test_swipe_gesture_detection():
    engine = GestureEngine(swipe_threshold=0.08)

    # Simulate fast horizontal movement across 6 consecutive frames
    for i in range(6):
        lms = np.zeros((21, 3), dtype=np.float32)
        lms[0] = [0.5, 0.8, 0.0]
        lms[9] = [0.5, 0.5, 0.0]
        # Index tip moving rapidly from left (0.1) to right (0.8)
        lms[8] = [0.1 + i * 0.12, 0.4, 0.0]
        lms[4] = [0.1 + i * 0.12, 0.6, 0.0]

        hand_data = [{
            "hand_id": 0,
            "handedness": "Right",
            "landmarks": lms,
            "landmarks_px": np.zeros((21, 2), dtype=np.int32),
            "palm_center": (320, 240),
            "palm_scale": 0.3
        }]

        time.sleep(0.01)
        res = engine.analyze(hand_data)

    # After sequence, swipe should trigger
    assert "SWIPE_RIGHT" in res["gestures"] or engine.last_swipe_time > 0

def test_mouse_fallback_timeout():
    engine = GestureEngine(fallback_timeout_frames=5)

    # Provide empty hand frames
    for _ in range(4):
        res = engine.analyze([])
        assert res["fallback_active"] is True or engine.no_hand_frames < 5

    # 5th frame reaches timeout
    res = engine.analyze([])
    assert res["fallback_active"] is True

    # When hand returns, fallback turns off
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = [0.5, 0.8, 0.0]
    lms[9] = [0.5, 0.5, 0.0]
    lms[8] = [0.5, 0.4, 0.0]
    lms[4] = [0.6, 0.5, 0.0]

    hand_data = [{
        "hand_id": 0,
        "handedness": "Right",
        "landmarks": lms,
        "landmarks_px": np.zeros((21, 2), dtype=np.int32),
        "palm_center": (320, 240),
        "palm_scale": 0.3
    }]

    res_active = engine.analyze(hand_data)
    assert res_active["fallback_active"] is False
