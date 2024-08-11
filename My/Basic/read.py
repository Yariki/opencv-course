import cv2 as cv

# img = cv.imread('../../Resources/Photos/cat_large.jpg')
# cv.imshow('cat', img)
# cv.waitKey(0)

#  Reading Videos

capture = cv.VideoCapture('../../Resources/Videos/dog.mp4') # int - 0 - present a webcam

while True:
    isTrue, frame = capture.read()

    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
