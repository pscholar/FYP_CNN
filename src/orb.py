import cv2
import numpy as np

# Load images
source = cv2.imread('bestref.png')
template = cv2.imread('input.jpg')

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(source, None)
kp2, des2 = orb.detectAndCompute(template, None)

# Create BFMatcher and find matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches based on distance
sorted_matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 10 matches
matching_result = cv2.drawMatches(source, kp1, template, kp2, sorted_matches[:10], None, flags=2)
cv2.imshow("Matches", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()