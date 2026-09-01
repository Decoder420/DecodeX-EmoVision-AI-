import numpy as np
import tensorflow as tf
from src.models import (
    build_cnn_model,
    build_mobilenet_model,
    compile_model,
    preprocess_face_roi,
    get_rescaling_layer,
    EMOTION_DICT
)

def test_cnn_architecture_and_forward_pass():
    model = build_cnn_model(input_shape=(48, 48, 1), num_classes=7, with_augmentation=False)
    assert model.name == "Emotion_CNN_4Block"
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, 7)

    # Forward pass with random batch
    dummy_input = np.random.uniform(0.0, 1.0, size=(4, 48, 48, 1)).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (4, 7)
    
    # Softmax probabilities should sum to ~1.0
    row_sums = np.sum(output.numpy(), axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0, 1.0], rtol=1e-5)

def test_mobilenet_architecture():
    model = build_mobilenet_model(input_shape=(48, 48, 1), num_classes=7, pretrained=False)
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, 7)

    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 48, 48, 1)).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (2, 7)

def test_preprocess_face_roi_normalization():
    # Create raw uint8 grayscale crop
    raw_crop = np.random.randint(0, 256, size=(100, 120), dtype=np.uint8)
    preprocessed = preprocess_face_roi(raw_crop)

    assert preprocessed.shape == (1, 48, 48, 1)
    assert preprocessed.dtype == np.float32
    assert np.min(preprocessed) >= 0.0
    assert np.max(preprocessed) <= 1.0

def test_rescaling_layer():
    rescaling = get_rescaling_layer()
    raw_tensor = tf.constant([[[[0.0]], [[255.0]]]])
    scaled = rescaling(raw_tensor)
    np.testing.assert_allclose(scaled.numpy(), [[[[0.0]], [[1.0]]]])
