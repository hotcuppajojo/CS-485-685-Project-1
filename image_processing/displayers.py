# displayers.py

# displayers.py

import cv2
import numpy as np

def display_img(image):
    # Check if the image is valid
    if image is None:
        raise ValueError("No image to display. Image input is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")

    if image.ndim not in [2, 3]:
        raise ValueError("Input must be a 2D (grayscale) or 3D (color) image.")

    # Display the image
    cv2.imshow('Image', image)

    # Wait for any key to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()