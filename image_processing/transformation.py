# transformations.py

import numpy as np

def rotate(image, theta, center=None):
    # Get the image dimensions
    h, w = image.shape[:2]

    # If no rotation center is specified, use the center of the image
    if center is None:
        center = (w // 2, h // 2)

    # Calculate the new image dimensions
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    new_w = int(abs(h * sin_theta) + abs(w * cos_theta))
    new_h = int(abs(w * sin_theta) + abs(h * cos_theta))

    # Create an empty output image
    output = np.zeros((new_h, new_w), dtype=image.dtype)

    # Calculate the offset between the new image and the old image
    offset_x = (new_w - w) // 2
    offset_y = (new_h - h) // 2

    # Calculate the inverse rotation matrix to map the output pixels to the input pixels
    inv_rotate_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

    # Apply the rotation transformation
    for i in range(new_h):
        for j in range(new_w):
            # Coordinate in the output image
            new_coord = np.array([j - center[0] - offset_x, i - center[1] - offset_y])
            # Calculate the corresponding pixel in the input image using the inverse matrix
            old_coord = np.dot(inv_rotate_matrix, new_coord) + np.array(center)

            # Bilinear interpolation
            x, y = old_coord
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)

            # Calculate the interpolation weights
            dx, dy = x - x0, y - y0
            weight_tl = (1 - dx) * (1 - dy)
            weight_tr = dx * (1 - dy)
            weight_bl = (1 - dx) * dy
            weight_br = dx * dy

            # If the calculated pixel lies within the bounds of the input image, interpolate its value
            if 0 <= x0 < w and 0 <= y0 < h:
                tl = image[y0, x0]
                tr = image[y0, x1]
                bl = image[y1, x0]
                br = image[y1, x1]

                # Calculate the interpolated pixel value
                interpolated_value = (tl * weight_tl + tr * weight_tr + bl * weight_bl + br * weight_br)
                output[i, j] = np.clip(interpolated_value, 0, 255)

    return output