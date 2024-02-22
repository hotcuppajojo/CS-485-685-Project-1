# transformations.py

import numpy as np

def rotate(image, theta, center=None):
    # Get the image dimensions
    h, w = image.shape[:2]

    # If no rotation center is specified, use the center of the image
    if center is None:
        center = (w / 2 - 0.5, h / 2 - 0.5)

    # Calculate the new image dimensions
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Compute the new bounding box size
    new_w = int(abs(h * sin_theta) + abs(w * cos_theta))
    new_h = int(abs(w * sin_theta) + abs(h * cos_theta))

    # Create an empty output image
    output = np.zeros((new_h, new_w), dtype=image.dtype)

    # Offset to translate the rotation center to the origin of the new image
    offset_x = (new_w - w) / 2
    offset_y = (new_h - h) / 2

    # Calculate the inverse rotation matrix to map the output pixels to the input pixels
    inv_rotate_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

    # Apply the rotation transformation
    for i in range(new_h):
        for j in range(new_w):
            # Translate pixel position to rotate around the center
            x = j - offset_x - center[0]
            y = i - offset_y - center[1]

            # Apply the rotation
            xp, yp = np.dot(inv_rotate_matrix, [x, y])

            # Translate back and round to nearest pixel
            xp += center[0]
            yp += center[1]

            # Interpolation
            if 0 <= xp < w and 0 <= yp < h:
                x0 = int(np.floor(xp))
                y0 = int(np.floor(yp))
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)

                fx = xp - x0
                fy = yp - y0

                # Interpolate between the four surrounding pixels
                a = image[y0, x0] * (1 - fx) * (1 - fy)
                b = image[y0, x1] * fx * (1 - fy)
                c = image[y1, x0] * (1 - fx) * fy
                d = image[y1, x1] * fx * fy

                # Calculate the interpolated pixel value
                output[i, j] = np.clip(a + b + c + d, 0, 255)
                
    return output