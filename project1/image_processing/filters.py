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
    stride = 1  # This is the stride of your convolution
    filter_height, filter_width = filter.shape

    # Calculate padding for height and width separately
    pad_height = pad_pixels if isinstance(pad_pixels, int) else pad_pixels[0]
    pad_width = pad_pixels if isinstance(pad_pixels, int) else pad_pixels[1]

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)),
                          mode='constant' if pad_value == 0 else 'edge',
                          constant_values=pad_value)

    # Calculate the dimensions of the output image
    output_height = ((image.shape[0] - filter_height + 2 * pad_height) // stride) + 1
    output_width = ((image.shape[1] - filter_width + 2 * pad_width) // stride) + 1
    output = np.zeros((output_height, output_width))

    # Apply the filter
    for i in range(0, output_height):
        for j in range(0, output_width):
            # Define the current region of interest
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + filter_height
            end_j = start_j + filter_width
            window = padded_image[start_i:end_i, start_j:end_j]

            # Calculate the convolution result (element-wise multiplication and sum)
            output_value = np.sum(window * filter)

            # Store the result
            output[i, j] = output_value

    # Normalize the output to the range [0, 255]
    output = ((output - np.min(output)) / (np.max(output) - np.min(output)) * 255)
    output = np.clip(output, 0, 255).astype(np.uint8)

    # Return the output
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