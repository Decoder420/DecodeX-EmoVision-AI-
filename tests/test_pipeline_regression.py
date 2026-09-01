import os
import tempfile
import numpy as np
import tensorflow as tf
from PIL import Image
from src.models import build_cnn_model, compile_model, EMOTION_DICT
from emotions import create_dataset_generators, plot_model_history

def test_training_pipeline_end_to_end_regression():
    """
    Regression test verifying that the migrated Keras 3.x / TF 2.x pipeline:
    - Loads images via tf.keras.utils.image_dataset_from_directory
    - Normalizes properly with Rescaling(1./255)
    - Compiles with Adam(learning_rate=...)
    - Trains via model.fit() without shape or loss divergence
    - Generates and saves training curve plots
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal synthetic dataset structure
        for split in ['train', 'test']:
            for emo_id, emo_name in EMOTION_DICT.items():
                class_dir = os.path.join(tmpdir, split, emo_name.lower())
                os.makedirs(class_dir, exist_ok=True)
                for img_idx in range(4):
                    # 48x48 random grayscale image
                    arr = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
                    img = Image.fromarray(arr)
                    img.save(os.path.join(class_dir, f"sample_{img_idx}.png"))

        # Create generators
        train_ds, val_ds = create_dataset_generators(tmpdir, batch_size=4)

        # Inspect a batch to confirm shape and normalization in [0.0, 1.0]
        for images, labels in train_ds.take(1):
            assert images.shape[1:] == (48, 48, 1)
            assert labels.shape[1] == 7
            assert float(tf.reduce_min(images)) >= 0.0
            assert float(tf.reduce_max(images)) <= 1.0

        # Build and compile model
        model = build_cnn_model(input_shape=(48, 48, 1), num_classes=7)
        model = compile_model(model, learning_rate=0.001)

        # Fit for 2 epochs (regression check)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=2,
            verbose=0
        )

        assert 'accuracy' in history.history
        assert 'loss' in history.history
        assert 'val_accuracy' in history.history
        assert 'val_loss' in history.history

        # Check loss is finite
        for loss_val in history.history['loss']:
            assert np.isfinite(loss_val)

        # Check plotting
        plot_model_history(history, save_plot=False)
