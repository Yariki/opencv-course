import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Calibration data (for demonstration, replace with your actual data)
data = pd.DataFrame({
    'Height': [1, 2, 3, 4, 5],  # Replace with actual height data
    'OpticalFlow': [10, 8, 6, 5, 3]  # Replace with actual optical flow data
})

# Prepare the data for model fitting
X = data['OpticalFlow'].values.reshape(-1, 1)
y = data['Height'].values

# Fit a polynomial regression model (e.g., degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Now, the video processing and height estimation
cap = cv2.VideoCapture(0)  # Use appropriate video source

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Capture the first frame and convert it to grayscale
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial points to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Calculate the average displacement (optical flow magnitude)
    if len(good_new) > 0:
        flow_magnitudes = np.linalg.norm(good_new - good_old, axis=1)
        avg_optical_flow = np.mean(flow_magnitudes)

        # Predict the height using the pre-trained model
        X_real_time = np.array([[avg_optical_flow]])
        X_real_time_poly = poly.transform(X_real_time)
        estimated_height = model.predict(X_real_time_poly)

        print(f"Estimated Height: {estimated_height[0]:.2f} meters")

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Display the video feed
    cv2.imshow('Drone Camera', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
