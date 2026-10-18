"""
Model loading, image preprocessing and prediction.
"""

import io
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CLASS_LABELS = ["Glioma", "Meningioma", "No tumor", "Pituitary"]

MODEL_CONFIGS = {
    "Xception (transfer learning)": {
        "file": "xception_model.weights.h5",
        "img_size": (299, 299),
    },
    "Custom CNN": {
        "file": "cnn_model.h5",
        "img_size": (224, 224),
    },
}


def build_xception_model(weights_path):
    img_shape = (299, 299, 3)

    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights=None,
        input_shape=img_shape,
        pooling="max",
    )

    model = Sequential([
        base_model,
        Dropout(rate=0.3),
        Dense(128, activation="relu"),
        Dropout(rate=0.25),
        Dense(4, activation="softmax"),
    ])

    model.build((None,) + img_shape)
    model.load_weights(weights_path)
    return model


def load_cnn_model(model_path):
    return load_model(model_path, compile=False)


def load_model_by_name(model_name):
    config = MODEL_CONFIGS[model_name]
    path = os.path.join(MODELS_DIR, config["file"])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if model_name.startswith("Xception"):
        return build_xception_model(path)

    
    return load_cnn_model(path)

def preprocess_image(image_bytes, img_size):
    img = load_img(io.BytesIO(image_bytes), target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict(model, img_array):
    probabilities = model.predict(img_array, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    label = CLASS_LABELS[class_index]
    confidence = float(probabilities[class_index])
    return label, confidence, probabilities