# test_edge_detection.py

import unittest
import numpy as np
from image_processing import edge_detection

class TestEdgeDetection(unittest.TestCase):
    def setUp(self):
        # Create a test image with a simple edge
        self.test_img = np.zeros((10, 10), dtype=np.uint8)
        self.test_img[:, 5:] = 255

    def test_edge_detection_vertical_edge(self):
        # Detecting edge on an image with a clear vertical edge
        edge_img = edge_detection(self.test_img)
        # We expect a strong response (high values) along the edge
        edge_strength = edge_img[:, 4:6].sum()
        self.assertTrue(edge_strength > 0, "Edge detection should respond to a vertical edge.")

    def test_edge_detection_no_edge(self):
        # Edge detection on a uniform image should yield low response
        uniform_img = np.zeros((10, 10), dtype=np.uint8)
        edge_img = edge_detection(uniform_img)
        self.assertTrue(np.max(edge_img) == 0, "Edge detection should have no response on a uniform image.")

    def test_edge_detection_invalid_image(self):
        # Edge detection should raise an error with invalid input
        with self.assertRaises(ValueError):
            edge_detection(np.array([1, 2, 3]))  # Not a 2D array

if __name__ == '__main__':
    unittest.main()