
# ============================================================
# AGRIMIND CROP CLASSIFIER
# ============================================================

import os
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocessing import preprocess_image


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

WHEAT_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "wheat_efficientnet_stage1_best.keras"
)

MAIZE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "maize_efficientnet_stage1_best_v2.keras"
)


# ============================================================
# CLASS NAMES
# ============================================================

WHEAT_CLASSES = [
    "Healthy",
    "septoria",
    "stripe_rust"
]

MAIZE_CLASSES = [
    "Healthy",
    "MLN",
    "MSV"
]


# ============================================================
# LOAD MODELS
# ============================================================

wheat_classifier = None
maize_classifier = None


def load_classification_models():

    global wheat_classifier
    global maize_classifier

    if os.path.exists(WHEAT_MODEL_PATH):

        wheat_classifier = load_model(
            WHEAT_MODEL_PATH
        )

    if os.path.exists(MAIZE_MODEL_PATH):

        maize_classifier = load_model(
            MAIZE_MODEL_PATH
        )


load_classification_models()


# ============================================================
# NORMALIZE DISEASE NAME
# ============================================================

def normalize_disease_name(disease):

    disease = disease.strip()

    replacements = {
        "stripe_rust": "Wheat Stripe Rust",
        "septoria": "Septoria",
        "MLN": "Maize Lethal Necrosis",
        "MSV": "Maize Streak Virus",
        "Healthy": "Healthy"
    }

    return replacements.get(
        disease,
        disease.replace("_", " ").title()
    )


# ============================================================
# CLASSIFY CROP
# ============================================================

def classify_crop(image_path, crop):

    crop = crop.lower().strip()


    if crop == "wheat":

        model = wheat_classifier
        class_names = WHEAT_CLASSES

    elif crop == "maize":

        model = maize_classifier
        class_names = MAIZE_CLASSES

    else:

        raise ValueError(
            "Unsupported crop. Use 'wheat' or 'maize'."
        )


    if model is None:

        raise FileNotFoundError(
            f"The {crop} classification model was not "
            f"found in: {MODEL_DIR}"
        )


    if len(class_names) != model.output_shape[-1]:

        raise ValueError(
            f"{crop} model has "
            f"{model.output_shape[-1]} outputs, "
            f"but {len(class_names)} class names "
            f"are configured."
        )


    image_array = preprocess_image(
        image_path
    )


    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )


    return {
        "crop": crop,
        "disease": predicted_class,
        "confidence": confidence,
        "probabilities": predictions
    }
