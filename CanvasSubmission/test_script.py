import cv2
import PIL
import matplotlib
import skimage
import numpy as np
import math
import project1 as p1

filter_size = 5
sigma = 1
theta = math.pi/2

#Load and display image
img = p1.load_img("test_img.jpg", True)
p1.display_img(img)

#Generate 1D gaussian filter
gaussian1D = p1.generate_gaussian(sigma, filter_size, 1)

#Filter image with 1D gaussian
gaussian1D_img = p1.apply_filter(img, gaussian1D, 0, 0)
p1.display_img(gaussian1D_img)

#Generate 2D gaussian filter
gaussian2D = p1.generate_gaussian(sigma, filter_size, filter_size)

#Filter image with 2D gaussian
gaussian2D_img = p1.apply_filter(img, gaussian2D, 0, 0)
p1.display_img(gaussian2D_img)

#Noise removal with median filter
median_img = p1.median_filtering(img, filter_size, filter_size)
p1.display_img(median_img)

#Rotate Image
transformed_img = p1.rotate(img, theta)
p1.display_img(transformed_img)

#Edge Detection
edges_img = p1.edge_detection(img)
p1.display_img(edges_img)

#Histogram Equalization Grayscale
eq_img = p1.hist_eq(img)
p1.display_img(eq_img)

img = p1.load_img("test_img.jpg", False)
p1.display_img(img)

#Histogram Equalization Color
eq_img = p1.hist_eq(img)
p1.display_img(eq_img)