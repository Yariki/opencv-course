import cv2 as cv

# img = cv.imread('../../Resources/Photos/cat_large.jpg')
# cv.imshow('Cat', img)
# cv.waitKey(0)

def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # Live video
    capture.set(3, width)
    capture.set(4, height)


capture = cv.VideoCapture('../../Resources/Videos/dog.mp4') # int - 0 - present a webcam

while True:
    isTrue, frame = capture.read()

    rescaled_frame = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Video Rescaled', rescaled_frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()


