# histogram.py

import numpy as np

def hist_eq(image):
    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        # Grayscale image
        # Calculate the histogram
        hist, bins = np.histogram(image.flatten(), 256, [0,256])

        # Calculate the cumulative distribution function manually
        cdf = np.zeros_like(hist)
        cdf[0] = hist[0]
        for i in range(1, len(hist)):
            cdf[i] = cdf[i-1] + hist[i]

        # Normalize the CDF
        cdf_normalized = cdf * float(hist.max()) / cdf.max()

        # Use the normalized CDF to map the pixels in the input image to their new, equalized values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)

        # Reshape the equalized image to the same shape as the input image
        image_equalized = image_equalized.reshape(image.shape)
    else:
        # Color image
        # Apply histogram equalization to each color channel separately
        image_equalized = np.zeros_like(image)
        for i in range(3):
            channel = image[:, :, i]

            # Calculate the histogram
            hist, bins = np.histogram(channel.flatten(), 256, [0,256])

            # Calculate the cumulative distribution function manually
            cdf = np.zeros_like(hist)
            cdf[0] = hist[0]
            for i in range(1, len(hist)):
                cdf[i] = cdf[i-1] + hist[i]

            # Normalize the CDF
            cdf_normalized = cdf * float(hist.max()) / cdf.max()

            # Use the normalized CDF to map the pixels in the input image to their new, equalized values
            channel_equalized = np.interp(channel.flatten(), bins[:-1], cdf_normalized)

            # Reshape the equalized channel to the same shape as the input channel
            image_equalized[:, :, i] = channel_equalized.reshape(channel.shape)

    return image_equalized