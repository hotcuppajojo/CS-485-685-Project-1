# test_loaders.py

import unittest
import cv2
from image_processing import load_img

class TestLoaders(unittest.TestCase):
    def test_load_image_grayscale(self):
        image = load_img('test_img.jpg', grayscale=True)
        self.assertEqual(len(image.shape), 2, "Image should be loaded in grayscale.")

    def test_load_image_color(self):
        image = load_img('test_img.jpg', grayscale=False)
        self.assertEqual(len(image.shape), 3, "Image should be loaded in color.")

    def test_load_image_nonexistent(self):
        with self.assertRaises(IOError):
            load_img('nonexistent.jpg')

if __name__ == '__main__':
    unittest.main()
