import  cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


def estimate_height(pixel_change_speed):
    # Placeholder for the calibration model
    # You need to replace this with a real model based on your calibration data
    height = pixel_change_speed #some_function_of(pixel_change_speed)
    return height


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    pixel_change_speed  = np.sum(diff)

    prev_gray = gray

    # Estimate the height
    height = estimate_height(pixel_change_speed)

    cv2.putText(frame, f'Height: {height:.2f} meters', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    cv2.imshow("Difference", diff)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cap.destroyAllWindows()