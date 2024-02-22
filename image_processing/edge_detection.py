# edge_detection.py

import numpy as np
from .filters import apply_filter

def edge_detection(image):
    # Check if the input image is a 2D array
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    
    # Define the Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply the Sobel filters to get the gradient in the x and y directions
    gradient_x = apply_filter(image, sobel_x, 1, 0)
    gradient_y = apply_filter(image, sobel_y, 1, 0)

    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Ensure gradient_magnitude is always defined by initializing it before any conditions that may skip its definition
    if gradient_magnitude.max() == 0:
        # Handle the case where the maximum gradient magnitude is 0 to avoid division by zero
        gradient_magnitude = np.zeros_like(image)
    else:
        # Normalize the gradient magnitude image to the range 0-255
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

    return gradient_magnitude