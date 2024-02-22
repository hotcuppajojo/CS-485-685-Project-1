# Project 1: Image Filtering by JoJo Petersky

## Usage
This project provides a comprehensive suite of image filtering functionalities, including loading and displaying images, applying Gaussian and median filters, performing histogram equalization, image transformation, and edge detection. To use these functionalities:

1. **Loading Images**: Use `load_img(file_name, grayscale=True)` to load images. The `grayscale` flag should be set to `True` for grayscale images and `False` for color images. This modification from the original assignment parameters allows for explicit control over image color space upon loading.

2. **Displaying Images**: Intermediate steps and results are displayed using `display_img(image)`. This enhancement to `test_script.py` enables visualization of images as each processing step is applied, facilitating a better understanding of the effects of each operation.

3. **Histogram Equalization Tests**: Histogram equalization functionalities have been positioned towards the end of the test script. This adjustment showcases the distinct outcomes of histogram equalization on grayscale versus color images more effectively. For color images, the test image is reloaded as a color image to emphasize the differences in processing and results between the two modes.

Ensure that the test script (`test_script.py`) is configured correctly to reflect these usage notes, particularly the modifications for image loading and the sequence of histogram equalization tests.

## 1. Load and Display Images

### Load Image Functionality
Utilizes OpenCV (`cv2`) for its efficient handling and broad support in image processing to load images. This choice streamlines the workflow for both grayscale and color images, directly aligning with processing requirements without significant overhead.

### Display Image Functionality
Employs `cv2.imshow` for its straightforward integration with OpenCV's loading functions and compatibility across various environments, ensuring a seamless display process.

## 2. 1D and 2D Filtering

### Gaussian Filter Generation
Implements the `generate_gaussian` function to create Gaussian filters, supporting both 1D and 2D configurations. This facilitates an understanding of Gaussian filtering effects, governed by the `sigma` parameter.

### Apply Filter Functionality
The `apply_filter` function performs convolution with specified padding strategies, allowing exploration of filtering outcomes at image boundaries. It demonstrates the impact of zero-padding versus edge-padding.

### Median Filtering
Applies a median filter to reduce noise while preserving edges, crucial for maintaining image quality and preparing images for further processing.

## 3. Histogram Equalization

Addresses histogram equalization for both grayscale and color images, using YCbCr color space for color images to maintain color balance while enhancing contrast. This method ensures enhanced visibility and detail without altering the color dynamics of the original image.

## 4. Image Transformation

### Rotation
Utilizes an inverse rotation matrix and bilinear interpolation to ensure spatial consistency and quality in rotated images. This method carefully balances computational efficiency with image integrity.

## 5. Edge Detection

### Overview
Integrates median filtering, Sobel gradient calculation, Otsu's thresholding, and hysteresis. This approach is designed to handle edge detection with high specificity and sensitivity, accommodating different image characteristics.

### Challenges and Solutions
- **Algorithmic Complexity and Overflows**: Encountered numerical stability issues, especially with gradient magnitude calculations in high-contrast regions, leading to potential overflows. Addressed by using higher precision data types (e.g., `np.float64`) during calculations and clipping values post-normalization to maintain valid pixel ranges.
- **Test Image Characteristics**: High-contrast edges, noise, and complex textures presented significant challenges. Implemented robust preprocessing (e.g., noise reduction) and dynamic thresholding (Otsu's method) to adapt to varying image features. Comprehensive testing across a diverse set of images ensured the robustness of the edge detection algorithm.

### Color Space Conversion in Histogram Equalization
- **Challenge**: The conversion to YCbCr color space for histogram equalization introduced complexities in accurately maintaining color integrity while enhancing contrast.
- **Solution**: Adopted a precise conversion process, ensuring luminance channel (Y) equalization without distorting the color components (Cb and Cr), followed by careful re-conversion to RGB. This approach provided a balanced enhancement, preserving the original color dynamics.