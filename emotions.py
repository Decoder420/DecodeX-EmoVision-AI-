import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.models import build_cnn_model, build_mobilenet_model, compile_model, EMOTION_DICT
from src.face_detector import FaceDetector
from src.inference import EmotionEngine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_arguments():
    ap = argparse.ArgumentParser(description="Emotion Detection using Deep Learning")
    ap.add_argument("--mode", default="display", choices=["train", "display"], help="train / display")
    ap.add_argument("--model", default="cnn", choices=["cnn", "mobilenet"], help="Model architecture (cnn or mobilenet)")
    ap.add_argument("--detector", default="mediapipe", choices=["mediapipe", "haar"], help="Face detector backend")
    ap.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    ap.add_argument("--lr", type=float, default=0.0001, help="Learning rate for Adam optimizer")
    ap.add_argument("--model_path", default="model.h5", help="Path to save or load model weights")
    ap.add_argument("--augment", action="store_true", help="Enable data augmentation during training")
    ap.add_argument("--data_dir", default="data", help="Directory containing train and test folders")
    return ap.parse_args()

def plot_model_history(model_history, save_plot=True):
    """
    Plot and save Accuracy and Loss curves given the model_history.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    epochs_range = range(1, len(model_history.history['accuracy']) + 1)

    # Accuracy Plot
    axs[0].plot(epochs_range, model_history.history['accuracy'], label='Train Accuracy', color='#2563eb', lw=2)
    if 'val_accuracy' in model_history.history:
        axs[0].plot(epochs_range, model_history.history['val_accuracy'], label='Val Accuracy', color='#16a34a', lw=2)
    axs[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right')
    axs[0].grid(True, alpha=0.3)

    # Loss Plot
    axs[1].plot(epochs_range, model_history.history['loss'], label='Train Loss', color='#dc2626', lw=2)
    if 'val_loss' in model_history.history:
        axs[1].plot(epochs_range, model_history.history['val_loss'], label='Val Loss', color='#ca8a04', lw=2)
    axs[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plot:
        fig.savefig('plot.png', dpi=300)
        fig.savefig('accuracy.png', dpi=300)
        print("[INFO] Plots saved to plot.png and accuracy.png")
    plt.close(fig)

def create_dataset_generators(data_dir, batch_size=64):
    """
    Creates normalized training and validation datasets using modern tf.keras.utils.image_dataset_from_directory.
    Falls back gracefully if data folders are missing.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Training/Testing data directories not found in '{data_dir}'. Run dataset_prepare.py first.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(48, 48),
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(48, 48),
        shuffle=False
    )

    # Normalize inputs to [0.0, 1.0]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds

def run_training(args):
    """
    Executes model training with modern Keras fit API, callbacks, and visualization.
    """
    print(f"[INFO] Initializing {args.model.upper()} model for training...")
    if args.model == "mobilenet":
        model = build_mobilenet_model(input_shape=(48, 48, 1), num_classes=7, with_augmentation=args.augment)
    else:
        model = build_cnn_model(input_shape=(48, 48, 1), num_classes=7, with_augmentation=args.augment)

    # Compile with modern Adam learning_rate argument
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    train_ds, val_ds = create_dataset_generators(args.data_dir, batch_size=args.batch_size)

    callbacks = [
        ModelCheckpoint(args.model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]

    print(f"[INFO] Starting training for {args.epochs} epochs...")
    # Modern model.fit instead of deprecated model.fit_generator
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    plot_model_history(history)
    model.save_weights(args.model_path)
    print(f"[INFO] Training complete. Model weights saved to '{args.model_path}'.")

def run_display(args):
    """
    Real-time webcam inference loop with MediaPipe/Haar Cascade face detection and normalized model prediction.
    """
    engine = EmotionEngine(
        model_path=args.model_path,
        model_type=args.model,
        detector_type=args.detector,
        cascade_path="haarcascade_frontalface_default.xml"
    )

    # Disable OpenCL if required for OpenCV stability
    cv2.ocl.setUseOpenCL(False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0). If you don't have a webcam connected, use the Web UI image upload mode.")
        return

    print("[INFO] Starting webcam feed. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from webcam.")
            break

        annotated_frame, results = engine.process_frame(frame, draw_annotations=True)

        display_frame = cv2.resize(annotated_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Emotion Detection (Press q to quit)', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.detector.close()

def is_running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if is_running_in_streamlit():
    from app import main as run_streamlit_app
    run_streamlit_app()
elif __name__ == '__main__':
    args = parse_arguments()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "display":
        run_display(args)