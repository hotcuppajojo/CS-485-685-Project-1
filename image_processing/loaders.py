# loaders.py

import cv2

def load_img(file_name):
    # Load the image
    img = cv2.imread(file_name)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img