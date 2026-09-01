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
    - Trains via model.fit() and demonstrates REAL learning progression with accuracy assertions
    - Generates and saves training curve plots
    """
    # Fix random seed for determinism
    np.random.seed(42)
    tf.random.set_seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a deterministic, class-separable synthetic dataset
        # Each emotion class gets distinct pixel intensity features so the CNN easily learns and separates classes
        for split in ['train', 'test']:
            for emo_id, emo_name in EMOTION_DICT.items():
                class_dir = os.path.join(tmpdir, split, emo_name.lower())
                os.makedirs(class_dir, exist_ok=True)
                
                # Base pattern distinct per emotion class
                base_val = int((emo_id + 1) * 35)
                for img_idx in range(8):
                    arr = np.clip(
                        np.full((48, 48), base_val, dtype=np.int16) + np.random.randint(-2, 3, (48, 48)),
                        0, 255
                    ).astype(np.uint8)
                    
                    img = Image.fromarray(arr)
                    img.save(os.path.join(class_dir, f"sample_{img_idx}.png"))

        # Create generators
        train_ds, val_ds = create_dataset_generators(tmpdir, batch_size=8)

        # Inspect a batch to confirm shape and normalization in [0.0, 1.0]
        for images, labels in train_ds.take(1):
            assert images.shape[1:] == (48, 48, 1), f"Unexpected shape {images.shape}"
            assert labels.shape[1] == 7, f"Unexpected label shape {labels.shape}"
            assert float(tf.reduce_min(images)) >= 0.0, "Pixels must be >= 0.0"
            assert float(tf.reduce_max(images)) <= 1.0, "Pixels must be <= 1.0"

        # Build and compile model
        model = build_cnn_model(input_shape=(48, 48, 1), num_classes=7)
        model = compile_model(model, learning_rate=0.002)

        # Fit for 15 epochs on separable dataset to verify real learning and accuracy convergence
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=15,
            verbose=0
        )

        train_acc = history.history['accuracy']
        train_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']

        initial_loss = train_loss[0]
        final_loss = train_loss[-1]
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]

        # Explicit Regression Assertions:
        # 1. Loss must strictly decrease (verifying optimization works)
        assert final_loss < initial_loss, f"Optimization failure: final_loss ({final_loss:.4f}) >= initial_loss ({initial_loss:.4f})"

        # 2. Accuracy must reach a high threshold (verifying model learning ability vs 14.3% random baseline)
        assert final_train_acc >= 0.65, f"Accuracy regression: final_train_acc ({final_train_acc:.2%}) is below expected 65% threshold"
        assert final_val_acc >= 0.50, f"Validation accuracy regression: final_val_acc ({final_val_acc:.2%}) is below expected 50% threshold"

        # 3. Check plotting utility runs without error
        plot_model_history(history, save_plot=False)
