# test_histogram.py

import unittest
import numpy as np
from image_processing import hist_eq

class TestHistogramEqualization(unittest.TestCase):
    def setUp(self):
        # Create a simple grayscale test image with a linear gradient
        self.grayscale_img = np.tile(np.arange(0, 256, dtype=np.uint8), (256, 1))

        # Create a simple color test image with a linear gradient
        self.color_img = np.dstack((self.grayscale_img, self.grayscale_img, self.grayscale_img))

    def test_hist_eq_grayscale(self):
        # Histogram equalization on a grayscale image
        equalized_img = hist_eq(self.grayscale_img)
        hist, _ = np.histogram(equalized_img, bins=256, range=[0, 256])
        
        # Check that the histogram is more uniform after equalization
        # This checks that no bin is excessively more populated than the average
        self.assertTrue(np.all(hist <= 1.2 * np.mean(hist)), "Histogram should be approximately uniform.")

        # Check that the cumulative distribution function is linear
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        self.assertTrue(np.all(np.abs(cdf_normalized - np.linspace(0, 1, 256)) < 0.05), "CDF should be approximately linear.")

    def test_hist_eq_color(self):
        # Histogram equalization on a color image
        equalized_img = hist_eq(self.color_img)
        
        # Check each channel separately
        for i in range(3):
            hist, _ = np.histogram(equalized_img[:, :, i], bins=256, range=[0, 256])
            # Check for uniformity
            self.assertTrue(np.all(hist <= 1.2 * np.mean(hist)), f"Histogram for channel {i} should be approximately uniform.")
            # Check for linearity of CDF
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]
            self.assertTrue(np.all(np.abs(cdf_normalized - np.linspace(0, 1, 256)) < 0.05), f"CDF for channel {i} should be approximately linear.")

    def test_hist_eq_invalid_input(self):
        # Test histogram equalization with invalid input (e.g., a 1D array)
        with self.assertRaises(ValueError):
            hist_eq(np.array([1, 2, 3]))

if __name__ == '__main__':
    unittest.main()
