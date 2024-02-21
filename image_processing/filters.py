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
    # Default stride is 1
    stride=1

    # Pad the image
    padded_image = np.pad(image, pad_pixels, mode='constant', constant_values=pad_value)

    # Calculate the dimensions of the output image
    output_height = ((image.shape[0] - filter.shape[0] + 2 * pad_pixels) // stride) + 1
    output_width = ((image.shape[1] - filter.shape[1] + 2 * pad_pixels) // stride) + 1
    output = np.zeros((output_height, output_width))

    # Apply the filter
    for i in range(0, image.shape[0] - filter.shape[0] + 1, stride):
        for j in range(0, image.shape[1] - filter.shape[1] + 1, stride):
            # Extract the window
            window = padded_image[i:i+filter.shape[0], j:j+filter.shape[1]]

            # Calculate the output value
            output_value = np.sum(window * filter)

            # Store the output value
            output[i // stride, j // stride] = output_value

    return output

def median_filtering(image, filter_w, filter_h):
    # Calculate the amount of padding needed
    pad_w, pad_h = filter_w // 2, filter_h // 2

    # Pad the image with the 'edge' mode which replicates the edge values
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Create an empty output image with the same shape as the input image
    output = np.zeros_like(image)

    # Apply the median filter
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the window with the correct offsets
            window = padded_image[i:i+filter_h, j:j+filter_w]

            # Replace the output pixel with the median of the window
            output[i, j] = np.median(window)

    return output