# loaders.py

import cv2

def load_img(file_name, grayscale=False):
    try:
        # Choose the color mode based on the grayscale flag
        color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

        # Attempt to load the image
        img = cv2.imread(file_name, color_mode)

        # Check if the image was loaded successfully
        if img is None:
            raise ValueError(f"The file {file_name} cannot be loaded as an image.")
        
        return img
    except Exception as e:
        # Handle other potential exceptions (e.g., file not found, no read permissions)
        raise IOError(f"An error occurred when trying to load the image: {e}")