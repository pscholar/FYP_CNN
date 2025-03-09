import cv2

# Load image and convert to grayscale
img = cv2.imread("bestref.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create FAST detector
fast = cv2.FastFeatureDetector_create()

# Detect corners
keypoints = fast.detect(gray, None)

# Draw keypoints
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

# Show the result
cv2.imshow("FAST Corners", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
