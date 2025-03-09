import cv2
import numpy as np

# Load the image
img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Apply Canny Edge Detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Step 2: Use Shi-Tomasi Corner Detector (only on edge regions)
corners = cv2.goodFeaturesToTrack(edges, maxCorners=500, qualityLevel=0.01, minDistance=20)

# Convert corner points to integer format
if corners is not None:
    corners = np.int0(corners)

    # Step 3: Draw detected corners
    for corner in corners:
        x, y = corner.ravel()  # Flatten array
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red circles for corners

# Show result
cv2.imshow("Canny + Shi-Tomasi Corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
