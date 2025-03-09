import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Harris Corner Detection
gray = np.float32(gray)  # Convert to float32
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate the detected corners to make them visible
dst = cv2.dilate(dst, None)

# Threshold to mark the corners in the original image
img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red

# Show the result
cv2.imshow("Harris Corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
