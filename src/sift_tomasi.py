import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def detect_corners(img):
    """Detect corners using Canny + Shi-Tomasi."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Apply Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Step 2: Detect Corners using Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(edges, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Convert to OpenCV keypoints
    keypoints = [cv2.KeyPoint(float(x), float(y), 1) for x, y in corners.reshape(-1, 2)] if corners is not None else []
    
    return keypoints, gray

def extract_descriptors(img, keypoints):
    """Extract Shape Context descriptors (approximated using SIFT)."""
    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(img, keypoints)
    return descriptors

def match_shapes(descriptors1, descriptors2):
    """Match shapes using the Hungarian Algorithm."""
    if descriptors1 is None or descriptors2 is None:
        return None, None  # No descriptors found
    
    # Compute pairwise distances between descriptors
    cost_matrix = cdist(descriptors1, descriptors2, metric='euclidean')

    # Solve assignment problem (Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return row_ind, col_ind

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    """Draw matched keypoints on the images."""
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, 
                                  [cv2.DMatch(i, j, 0) for i, j in zip(matches[0], matches[1])],
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_img

# Load images
img1 = cv2.imread("bestref.png")
img2 = cv2.imread("bestref.png")

# Detect corners
keypoints1, gray1 = detect_corners(img1)
keypoints2, gray2 = detect_corners(img2)

# Extract descriptors
descriptors1 = extract_descriptors(gray1, keypoints1)
descriptors2 = extract_descriptors(gray2, keypoints2)

# Match shapes
matches = match_shapes(descriptors1, descriptors2)

# Draw results
if matches[0] is not None:
    matched_img = draw_matches(img1, keypoints1, img2, keypoints2, matches)
    cv2.imshow("Shape Context Matching", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matches found.")
