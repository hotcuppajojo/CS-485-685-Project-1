# edge_detection.py

import numpy as np
from skimage import filters as skfilters
from . import filters

def edge_detection(image):
    # Check if the input image is a 2D array
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    
    # Apply median filtering to the image
    filterd_image = filters.median_filtering(image, 1, 1)

    # Define the Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply the Sobel filters to get the gradient in the x and y directions
    gradient_x = filters.apply_filter(filterd_image, sobel_x, 1, 0)
    gradient_y = filters.apply_filter(filterd_image, sobel_y, 1, 0)

    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the gradient magnitude to the range [0, 255]
    if gradient_magnitude.max() == 0:
        gradient_magnitude = np.zeros_like(image)
    else:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

    # Use Otsu's method to find a base threshold
    base_threshold = skfilters.threshold_otsu(gradient_magnitude)
    
    # Define lower and higher thresholds
    low_threshold = base_threshold * 0.5
    high_threshold = base_threshold * 1.5

    # Initialize matrices to identify strong, weak, and non-edges
    strong_edges = (gradient_magnitude >= high_threshold)
    weak_edges = ((gradient_magnitude >= low_threshold) & (gradient_magnitude < high_threshold))
    edges = np.zeros_like(image, dtype=bool)

    # Apply hysteresis
    # Mark strong edges as true immediately
    edges[strong_edges] = True
    
    # Iteratively propagate strong edges along weak edges
    height, width = gradient_magnitude.shape
    for row in range(1, height-1):
        for col in range(1, width-1):
            if weak_edges[row, col]:
                if np.any(strong_edges[row-1:row+2, col-1:col+2]):
                    edges[row, col] = True

    # Convert boolean edges to 8-bit format suitable for OpenCV display
    edges_display = (edges * 255).astype(np.uint8)

    return edges_display