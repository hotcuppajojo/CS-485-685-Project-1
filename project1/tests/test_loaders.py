# test_loaders.py

import unittest
from unittest.mock import patch
import numpy as np
from image_processing import load_img

class TestLoaders(unittest.TestCase):
    @patch('cv2.imread')
    def test_load_image_grayscale(self, mock_imread):
        # Mock imread to return a grayscale image
        mock_imread.return_value = np.zeros((10, 10), dtype=np.uint8)
        image = load_img('test_img.jpg', grayscale=True)
        self.assertEqual(len(image.shape), 2, "Image should be loaded in grayscale.")

    @patch('cv2.imread')
    def test_load_image_color(self, mock_imread):
        # Mock imread to return a color image
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        image = load_img('test_img.jpg', grayscale=False)
        self.assertEqual(len(image.shape), 3, "Image should be loaded in color.")

    @patch('cv2.imread')
    def test_load_image_nonexistent(self, mock_imread):
        # Mock imread to return None, simulating a failed load
        mock_imread.return_value = None
        with self.assertRaises(IOError):
            load_img('nonexistent.jpg')

if __name__ == '__main__':
    unittest.main()
