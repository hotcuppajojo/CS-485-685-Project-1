# test_transformation.py

import unittest
import numpy as np
from transformations import rotate

class TestTransformations(unittest.TestCase):
    def setUp(self):
        # Create a simple test image with known properties
        self.test_img = np.zeros((10, 10), dtype=np.uint8)
        self.test_img[4:6, 4:6] = 255

    def test_rotate_identity(self):
        # Rotating by 0 degrees should return the original image
        rotated_img = rotate(self.test_img, 0)
        np.testing.assert_array_equal(rotated_img, self.test_img)

    def test_rotate_quarter_turn(self):
        # Rotating a square image by 90 degrees should transpose the image
        rotated_img = rotate(self.test_img, np.pi/2)
        expected_img = np.rot90(self.test_img)
        np.testing.assert_array_equal(rotated_img, expected_img)

    def test_rotate_non_square_image(self):
        # Test rotation on a non-square image
        non_square_img = np.zeros((20, 10), dtype=np.uint8)
        rotated_img = rotate(non_square_img, np.pi/2)
        self.assertEqual(rotated_img.shape, (10, 20), "Rotated non-square image should have transposed dimensions.")

    def test_rotate_invalid_angle(self):
        # Test rotation with an invalid angle type (e.g., a string)
        with self.assertRaises(TypeError):
            rotate(self.test_img, '90')

if __name__ == '__main__':
    unittest.main()
