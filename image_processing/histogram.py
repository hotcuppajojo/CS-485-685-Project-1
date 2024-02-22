# histogram.py

import numpy as np

def rgb_to_ycbcr(image):
    """
    Convert an RGB image to YCbCr.
    """
    if image.dtype != np.float32:  # Convert to float32 for accurate calculations
        image = np.float32(image)
    
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
    
    YCbCr = np.stack((Y, Cb, Cr), axis=-1)
    return np.clip(YCbCr, 0, 255).astype(np.uint8)

def ycbcr_to_rgb(image):
    """
    Convert a YCbCr image back to RGB.
    """
    if image.dtype != np.float32:  # Convert to float32 for accurate calculations
        image = np.float32(image)
    
    Y, Cb, Cr = image[:, :, 0], image[:, :, 1] - 128, image[:, :, 2] - 128
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    
    RGB = np.stack((R, G, B), axis=-1)
    return np.clip(RGB, 0, 255).astype(np.uint8)


def _equalize_grayscale(image):
    """
    Apply histogram equalization to a grayscale (Y channel) image.
    """
    # Flatten the image and calculate the histogram
    img_flat = image.flatten()

    # Calculate the histogram and the cumulative distribution function (CDF)
    hist, _ = np.histogram(img_flat, bins=256, range=[0, 256])
    cdf = hist.cumsum()

    # Normalize the CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Apply the equalization
    img_equalized = cdf_normalized[img_flat].reshape(image.shape)

    return img_equalized.astype(np.uint8)

def _equalize_color(image):
    """
    Apply histogram equalization to the luminance channel of a color image in YCbCr color space.
    """
    # Convert RGB to YCbCr
    img_ycbcr = rgb_to_ycbcr(image)
    
    # Apply histogram equalization to the Y channel
    img_ycbcr[:,:,0] = _equalize_grayscale(img_ycbcr[:,:,0])
    
    # Convert back to RGB
    img_rgb_equalized = ycbcr_to_rgb(img_ycbcr)
    
    return img_rgb_equalized

def hist_eq(image):
    """
    Detect if an image is grayscale or color and apply histogram equalization accordingly.
    """
    if image.ndim == 2:  # Grayscale image
        return _equalize_grayscale(image)
    elif image.ndim == 3 and image.shape[2] == 3:  # Color image
        return _equalize_color(image)
    else:
        raise ValueError("Unsupported image format.")