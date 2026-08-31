
# ============================================================
# AGRIMIND IMAGE PREPROCESSING
# ============================================================

import numpy as np
import tensorflow as tf


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)


# ============================================================
# LOAD AND PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):
    """
    Load an image and prepare it for the trained
    EfficientNetB0 classifier.

    The training pipeline used 224x224 RGB images.
    EfficientNet preprocessing is already represented
    in the trained model graph, so no manual /255 scaling
    is applied here.
    """

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array
