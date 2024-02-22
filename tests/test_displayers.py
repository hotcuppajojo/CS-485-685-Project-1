# test_displayers.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from image_processing import display_img

class TestDisplayers(unittest.TestCase):
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_display_img(self, mock_destroy, mock_waitkey, mock_imshow):
        # Create a mock image as a numpy ndarray
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        display_img(test_img)
        
        mock_imshow.assert_called_once_with('Image', test_img)
        mock_waitkey.assert_called_once_with(0)
        mock_destroy.assert_called_once()

if __name__ == '__main__':
    unittest.main()