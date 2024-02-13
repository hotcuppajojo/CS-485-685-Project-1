# edge_detection.py

import numpy as np
from filters import apply_filter 

def edge_detection(image):
    # Define the Sobel filters
    def apply_filter(image, filter, stride, padding):
        # Implementation of the apply_filter function
        pass

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply the Sobel filters to get the gradient in the x and y directions
    gradient_x = apply_filter(image, sobel_x, 1, 0)
    gradient_y = apply_filter(image, sobel_y, 1, 0)

    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the gradient magnitude image to the range 0-255
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

    return gradient_magnitude