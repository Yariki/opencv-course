import  cv2  as cv
import numpy as np

img =  cv.imread('../../Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# Avaeraging
average = cv.blur(img, (7,7))
cv.imshow('Average Blur', average)


# gaussian blur
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian Blur', gauss)

# Median Blur
median = cv.medianBlur(img, 7)
cv.imshow('Median Blur', median)

# Bilateral Blur
bilateral = cv.bilateralFilter(img, 10, 25, 35)
cv.imshow('Bilateral Blur', bilateral)

cv.waitKey(0)