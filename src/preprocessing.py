"""
src/preprocessing.py
--------------------
All image preprocessing logic for the blood cell classification pipeline.
Used by src/prediction.py, api.py, and the retraining job.

Preprocessing contract:
  - Input : raw bytes  OR  numpy uint8 array (H, W, 3)  OR  PIL Image
  - Output: numpy float32 array ready for model inference
"""

import io
import cv2
import numpy as np
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = (96, 96)
CLASSES     = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
NUM_CLASSES = len(CLASSES)


# ── Core preprocessing functions ───────────────────────────────────────────────

def bytes_to_array(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes (from file upload) to a uint8 numpy array.

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes from an uploaded file.

    Returns
    -------
    np.ndarray
        uint8 array of shape (H, W, 3) in RGB order.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def resize(image: np.ndarray, size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Resize a uint8 image array to the target size using bilinear interpolation.

    Parameters
    ----------
    image : np.ndarray
        uint8 array of shape (H, W, 3).
    size : tuple
        Target (height, width).

    Returns
    -------
    np.ndarray
        uint8 array of shape (size[0], size[1], 3).
    """
    resized = cv2.resize(
        image,
        (size[1], size[0]),          # cv2 expects (width, height)
        interpolation=cv2.INTER_LINEAR
    )
    return resized.astype(np.uint8)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Scale pixel values from [0, 255] to [0.0, 1.0].
    Used for the Custom CNN model.

    Parameters
    ----------
    image : np.ndarray
        uint8 array of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        float32 array of shape (H, W, 3) with values in [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0


def augment(image: np.ndarray) -> np.ndarray:
    """
    Apply mild augmentation on raw uint8 pixels.
    Used during retraining to increase effective dataset size.

    Augmentations applied:
      - Horizontal flip (50% probability)
      - Vertical flip (50% probability)
      - Rotation ± 10 degrees
      - Contrast jitter ± 10%

    Parameters
    ----------
    image : np.ndarray
        uint8 array of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Augmented uint8 array of shape (H, W, 3).
    """
    import random

    if random.random() > 0.5:
        image = np.fliplr(image)
    if random.random() > 0.5:
        image = np.flipud(image)

    angle = random.uniform(-10, 10)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    factor = random.uniform(0.90, 1.10)
    image  = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return image


def prepare_for_inference(image_bytes: bytes) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image at inference time.
    Converts raw bytes to a model-ready batch tensor.

    Steps:
      1. Decode bytes to uint8 RGB array
      2. Resize to IMG_SIZE
      3. Normalise to [0, 1]
      4. Add batch dimension

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes from an uploaded image file.

    Returns
    -------
    np.ndarray
        float32 array of shape (1, 96, 96, 3) ready for model.predict().
    """
    arr     = bytes_to_array(image_bytes)
    arr     = resize(arr)
    arr     = normalize(arr)
    return np.expand_dims(arr, axis=0)


def prepare_batch_for_retraining(
    image_list: list,
    label_list: list,
    apply_augmentation: bool = True
) -> tuple:
    """
    Prepare a list of raw images and integer labels into a training batch.

    Parameters
    ----------
    image_list : list of np.ndarray
        List of uint8 arrays (any size, any channel order).
    label_list : list of int
        Integer class indices (0–3) matching CLASSES order.
    apply_augmentation : bool
        Whether to apply augmentation. True for training, False for evaluation.

    Returns
    -------
    tuple (X, y)
        X : float32 array of shape (N, 96, 96, 3)
        y : float32 one-hot array of shape (N, 4)
    """
    X = np.zeros((len(image_list), *IMG_SIZE, 3), dtype=np.float32)
    y = np.zeros((len(image_list), NUM_CLASSES),  dtype=np.float32)

    for i, (img, label) in enumerate(zip(image_list, label_list)):
        img = resize(img)
        if apply_augmentation:
            img = augment(img)
        X[i] = normalize(img)
        y[i, label] = 1.0

    return X, y
