import   cv2 as cv
import numpy as np

blank = np.zeros((400,400), dtype='uint8')

rect = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow('Rectangle', rect)
cv.imshow('Circle', circle)

# and - intersecting regions
bitwise_and = cv.bitwise_and(rect, circle)
cv.imshow('Bitwise AND', bitwise_and)


 # or - non and intersecting regions
bitwise_or = cv.bitwise_or(rect, circle)
cv.imshow('Bitwise OR', bitwise_or)

# xor - non intersecting regions
bitwise_xor = cv.bitwise_xor(rect, circle)
cv.imshow('Bitwise XOR', bitwise_xor)


# not - invert the binary color
bitwise_not = cv.bitwise_not(bitwise_xor)
cv.imshow('Bitwise NOT', bitwise_not)

cv.waitKey(0)
