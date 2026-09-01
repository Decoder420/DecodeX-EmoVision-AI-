import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D, Input, RandomFlip, RandomRotation, RandomZoom, RandomTranslation, Rescaling,
    UpSampling2D
)
from tensorflow.keras.optimizers import Adam

# Canonical FER-2013 emotion mapping
EMOTION_DICT = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

def get_rescaling_layer():
    """
    Single source of truth for input normalization layer (0-255 -> 0.0-1.0).
    """
    return Rescaling(1.0 / 255.0, name="rescaling_norm")

def preprocess_face_roi(roi_gray):
    """
    Single source of truth for preprocessing a single cropped grayscale face ROI.
    Ensures exact alignment with the training normalization pipeline.
    
    Args:
        roi_gray (np.ndarray): 2D grayscale face array
    Returns:
        np.ndarray: (1, 48, 48, 1) float32 array normalized to [0.0, 1.0]
    """
    resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    expanded = np.expand_dims(np.expand_dims(resized, -1), 0)
    normalized = expanded.astype(np.float32) / 255.0
    return normalized

def get_augmentation_layers():
    """
    Keras preprocessing layers for data augmentation to improve generalization.
    """
    return Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomTranslation(height_factor=0.05, width_factor=0.05)
    ], name="data_augmentation")

def build_cnn_model(input_shape=(48, 48, 1), num_classes=7, with_augmentation=False):
    """
    Standard 4-block CNN architecture for FER-2013 facial expression recognition.
    """
    model = Sequential(name="Emotion_CNN_4Block")
    model.add(Input(shape=input_shape))

    if with_augmentation:
        model.add(get_augmentation_layers())

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def build_mobilenet_model(input_shape=(48, 48, 1), num_classes=7, with_augmentation=False, pretrained=True):
    """
    Transfer learning baseline using MobileNetV3-Small.
    
    Trade-off Note:
    ImageNet pretraining expects 3-channel inputs (typically >=224x224). For FER-2013 (48x48 grayscale):
    - Channel adaptation: 1x1 conv projects 1 grayscale channel -> 3 channels.
    - Resolution adaptation: Optional 2x upsampling (to 96x96) or direct 48x48 processing.
    - Fine-tuning vs Scratch: If pretrained=False, trains MobileNet architecture directly on 48x48 without ImageNet prior.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    if with_augmentation:
        x = get_augmentation_layers()(x)

    # 1x1 projection from 1 grayscale channel -> 3 RGB channels
    x = Conv2D(3, kernel_size=(1, 1), padding='same', name="channel_expansion")(x)
    # Upsample to 96x96 to improve pretrained feature extraction
    x = UpSampling2D(size=(2, 2), name="resolution_upsample")(x)

    weights = 'imagenet' if pretrained else None
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(96, 96, 3),
        include_top=False,
        weights=weights,
        pooling=None
    )
    base_model.trainable = True

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Emotion_MobileNetV3Small")
    return model

def compile_model(model, learning_rate=0.0001):
    """
    Compiles model with Adam optimizer (using modern learning_rate param) and categorical crossentropy loss.
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model
