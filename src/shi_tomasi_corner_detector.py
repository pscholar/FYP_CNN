import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread("bestref.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Shi-Tomasi Corner Detector
corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)

# Convert corners to integer coordinates
corners = np.int0(corners)

# Draw detected corners
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw green circles

# Show the result
cv2.imshow("Shi-Tomasi Corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
