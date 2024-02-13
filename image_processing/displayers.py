# displayers.py

import cv2

def display_img(image):
    # Display the image
    cv2.imshow('Image', image)

    # Wait for any key to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()