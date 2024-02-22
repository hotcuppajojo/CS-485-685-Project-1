# histogram.py

import numpy as np

def hist_eq(image):
    # Validate input dimensions
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be either a 2D grayscale or 3D color image.")
    
    if image.ndim == 2:  # Grayscale image
        return _equalize_grayscale(image)
    else:  # Color image
        return _equalize_color(image)

def _equalize_grayscale(image):
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf.max()
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)
    return image_equalized.astype(np.uint8)

def _equalize_color(image):
    image_equalized = np.zeros_like(image)
    for i in range(3):  # Correctly iterate through each color channel
        image_equalized[:, :, i] = _equalize_grayscale(image[:, :, i])
    return image_equalized
