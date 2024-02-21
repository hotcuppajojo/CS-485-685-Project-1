# test_filters.py

import unittest
import numpy as np
from filters import apply_filter, generate_gaussian, median_filtering

class TestFilters(unittest.TestCase):
    def test_generate_gaussian(self):
        gaussian_filter = generate_gaussian(1.0, 3, 3)
        self.assertEqual(gaussian_filter.shape, (3, 3), "Gaussian filter should be 3x3.")

    def test_apply_filter(self):
        test_img = np.ones((5, 5))
        test_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filtered_img = apply_filter(test_img, test_filter, 1, 0)
        self.assertEqual(filtered_img.shape, test_img.shape, "Filtered image should have the same shape as the input image.")

    def test_median_filtering(self):
        test_img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        median_img = median_filtering(test_img, 3, 3)
        self.assertEqual(median_img[1, 1], 5, "Median should be the middle value.")

if __name__ == '__main__':
    unittest.main()