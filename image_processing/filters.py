# filters.py

import numpy as np

def generate_gaussian(sigma, filter_w, filter_h):
    m, n = [(ss-1.)/2. for ss in (filter_w, filter_h)]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def apply_filter(image, filter, pad_pixels, pad_value):
    # Pad the image
    padded_image = np.pad(image, pad_pixels, mode='constant', constant_values=pad_value)

    # Create an empty output image
    output = np.zeros_like(image)

    # Get the filter dimensions
    filter_w, filter_h = filter.shape

    # Apply the filter
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the window
            window = padded_image[i:i+filter_w, j:j+filter_h]

            # Replace the output pixel with the weighted sum of the window
            output[i, j] = np.sum(window * filter)

    return output

def median_filtering(image, filter_w, filter_h):
    # Pad the image
    padded_image = np.pad(image, ((filter_w//2, filter_w//2), (filter_h//2, filter_h//2)), mode='constant')

    # Create an empty output image
    output = np.zeros_like(image)

    # Apply the median filter
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the window
            window = padded_image[i:i+filter_w, j:j+filter_h]

            # Replace the output pixel with the median of the window
            output[i, j] = np.median(window)

    return output