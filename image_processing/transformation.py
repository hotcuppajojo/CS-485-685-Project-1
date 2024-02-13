# transformations.py

import numpy as np

def rotate(image, angle, center=None):
    # Get the image dimensions
    h, w = image.shape[:2]

    # If no rotation center is specified, use the center of the image
    if center is None:
        center = (w / 2, h / 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Create an empty output image
    output = np.zeros_like(image)

    # Apply the rotation transformation
    for i in range(h):
        for j in range(w):
            # Calculate the corresponding pixel in the input image
            x = (j - center[0]) * np.cos(angle) + (i - center[1]) * np.sin(angle) + center[0]
            y = -(j - center[0]) * np.sin(angle) + (i - center[1]) * np.cos(angle) + center[1]

            # If the calculated pixel lies within the bounds of the input image, copy its value to the output image
            if 0 <= x < w and 0 <= y < h:
                output[i, j] = image[int(y), int(x)]

    return output