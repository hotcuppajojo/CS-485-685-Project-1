# test_displayers.py

import unittest
from unittest.mock import patch
from displayers import display_img

class TestDisplayers(unittest.TestCase):
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_display_img(self, mock_destroy, mock_waitkey, mock_imshow):
        test_img = "path/to/test/image.jpg"  # Replace with actual path to an image file
        display_img(test_img)
        mock_imshow.assert_called_once()
        mock_waitkey.assert_called_once_with(0)
        mock_destroy.assert_called_once()

if __name__ == '__main__':
    unittest.main()
