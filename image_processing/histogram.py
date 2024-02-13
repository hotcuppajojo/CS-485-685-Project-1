# histogram.py

import numpy as np

def hist_eq(image):
    # Calculate the histogram
    hist, bins = np.histogram(image.flatten(), 256, [0,256])

    # Calculate the cumulative distribution function
    cdf = hist.cumsum()

    # Normalize the CDF
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Use the normalized CDF to map the pixels in the input image to their new, equalized values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    # Reshape the equalized image to the same shape as the input image
    image_equalized = image_equalized.reshape(image.shape)

    return image_equalized