import cv2 as cv
import numpy as np


img = cv.imread('../../Resources/Photos/park.jpg')
cv.imshow('Boston', img)

# Translation
# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

def translate(img, x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = img.shape[1], img.shape[0]
    return cv.warpAffine(img, transMat, dimensions)

translated = translate(img, 200, 200)
cv.imshow('Translated', translated)

# Rotation

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    # set a default value for the rotation point
    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotationMatrix = cv.getRotationMatrix2D(rotPoint, angle, 1.0)

    return cv.warpAffine(img, rotationMatrix, (width, height))

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

# Resizing
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)


# Flipping
# 0 --> flip vertically
# 1 --> flip horizontally
# -1 --> flip both vertically and horizontally
flip = cv.flip(img, 0)
cv.imshow('Flip', flip)


# Cropping
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)


cv.waitKey(0)