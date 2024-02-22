# JoJo Petersky
# CS 485/685 Spring '24 Project1
# 2024/2/20
# project1.py

import cv2
import numpy as np
from skimage import filters as skfilters

# Display Functions
# -----------------

def display_img(image):
    """
    Displays an image using OpenCV's imshow function.

    Parameters:
    - image: A numpy ndarray representing the image to be displayed.

    Raises:
    - ValueError: If the input image is None.
    - TypeError: If the input is not a numpy ndarray.
    - ValueError: If the input is not a 2D (grayscale) or 3D (color) image.
    """
    if image is None:
        raise ValueError("No image to display. Image input is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")

    if image.ndim not in [2, 3]:
        raise ValueError("Input must be a 2D (grayscale) or 3D (color) image.")

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Edge Detection Functions
# ------------------------

def edge_detection(image):
    """
    Performs edge detection on an input image using the Sobel operator and hysteresis thresholding.

    Parameters:
    - image: A 2D numpy array representing a grayscale image.

    Returns:
    - A numpy array with the detected edges.

    Raises:
    - ValueError: If the input image is not a 2D array.
    """
    # Check if the input image is a 2D array
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    
    # Apply median filtering to the image
    filterd_image = median_filtering(image, 1, 1)

    # Define the Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply the Sobel filters to get the gradient in the x and y directions
    gradient_x = apply_filter(filterd_image, sobel_x, 1, 0)
    gradient_y = apply_filter(filterd_image, sobel_y, 1, 0)

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

# Filter Functions
# ----------------

def generate_gaussian(sigma, filter_w, filter_h):
    """
    Generates a Gaussian filter kernel.

    Parameters:
    - sigma: Standard deviation of the Gaussian distribution.
    - filter_w: Width of the filter kernel.
    - filter_h: Height of the filter kernel.

    Returns:
    - A 2D numpy array representing the Gaussian filter.
    """
    m, n = [(ss-1.)/2. for ss in (filter_w, filter_h)]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def apply_filter(image, filter, pad_pixels, pad_value):
    """
    Applies a given filter to an image using convolution.

    Parameters:
    - image: The input image as a 2D numpy array.
    - filter: The filter kernel as a 2D numpy array.
    - pad_pixels: The amount of padding to apply to the image.
    - pad_value: The value to use for padding.

    Returns:
    - The filtered image as a 2D numpy array.
    """
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
    """
    Applies median filtering to an image.

    Parameters:
    - image: The input image as a 2D numpy array.
    - filter_w: Width of the median filter.
    - filter_h: Height of the median filter.

    Returns:
    - The filtered image as a 2D numpy array.
    """
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

# Histogram Equalization Functions
# ---------------------------------

def rgb_to_ycbcr(image):
    """
    Converts an RGB image to YCbCr color space.

    Parameters:
    - image: The input RGB image as a 3D numpy array.

    Returns:
    - The converted YCbCr image as a 3D numpy array.
    """
    # Convert to float32 for accurate calculations
    if image.dtype != np.float32:  
        image = np.float32(image)
    
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
    
    YCbCr = np.stack((Y, Cb, Cr), axis=-1)
    return np.clip(YCbCr, 0, 255).astype(np.uint8)

def ycbcr_to_rgb(image):
    """
    Converts a YCbCr image back to RGB color space.

    Parameters:
    - image: The input YCbCr image as a 3D numpy array.

    Returns:
    - The converted RGB image as a 3D numpy array.
    """
    # Convert to float32 for accurate calculations
    if image.dtype != np.float32:  
        image = np.float32(image)
    
    Y, Cb, Cr = image[:, :, 0], image[:, :, 1] - 128, image[:, :, 2] - 128
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    
    RGB = np.stack((R, G, B), axis=-1)
    return np.clip(RGB, 0, 255).astype(np.uint8)

def _equalize_grayscale(image):
    """
    Applies histogram equalization to a grayscale image.

    Parameters:
    - image: The input grayscale image as a 2D numpy array.

    Returns:
    - The histogram-equalized image as a 2D numpy array.
    """
    # Flatten the image and calculate the histogram
    img_flat = image.flatten()

    # Calculate the histogram and the cumulative distribution function (CDF)
    hist, _ = np.histogram(img_flat, bins=256, range=[0, 256])
    cdf = hist.cumsum()

    # Normalize the CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Apply the equalization
    img_equalized = cdf_normalized[img_flat].reshape(image.shape)

    return img_equalized.astype(np.uint8)

def _equalize_color(image):
    """
    Applies histogram equalization to the luminance channel of a color image in YCbCr color space.

    Parameters:
    - image: The input color image as a 3D numpy array.

    Returns:
    - The histogram-equalized RGB image as a 3D numpy array.
    """
    # Convert RGB to YCbCr
    img_ycbcr = rgb_to_ycbcr(image)
    
    # Apply histogram equalization to the Y channel
    img_ycbcr[:,:,0] = _equalize_grayscale(img_ycbcr[:,:,0])
    
    # Convert back to RGB
    img_rgb_equalized = ycbcr_to_rgb(img_ycbcr)
    
    return img_rgb_equalized

def hist_eq(image):
    """
    Detects if an image is grayscale or color and applies histogram equalization accordingly.

    Parameters:
    - image: The input image as a numpy array.

    Returns:
    - The histogram-equalized image.

    Raises:
    - ValueError: If the image format is unsupported.
    """
    if image.ndim == 2:  # Grayscale image
        return _equalize_grayscale(image)
    elif image.ndim == 3 and image.shape[2] == 3:  # Color image
        return _equalize_color(image)
    else:
        raise ValueError("Unsupported image format.")

# Image Loading Functions
# -----------------------

def load_img(file_name, grayscale=False):
    """
    Loads an image from a file.

    Parameters:
    - file_name: The path to the image file.
    - grayscale: A boolean flag indicating whether to load the image as grayscale.

    Returns:
    - The loaded image as a numpy array.

    Raises:
    - IOError: If the image cannot be loaded.
    """
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

# Image Transformation Functions
# ------------------------------

def rotate(image, theta, center=None):
    """
    Rotates an image around a specified center point.

    Parameters:
    - image: The input image as a numpy array.
    - theta: The rotation angle in radians.
    - center: The center of rotation (x, y). If None, uses the image center.

    Returns:
    - The rotated image as a numpy array.
    """
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
