import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def detect_corners(img):
    """Detect corners using Canny + Shi-Tomasi."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    corners = cv2.goodFeaturesToTrack(edges, maxCorners=100, qualityLevel=0.01, minDistance=10)
    return corners.reshape(-1, 2) if corners is not None else np.array([]), gray

def compute_tangent_angles(points):
    """Estimate local tangent angles using PCA."""
    n = len(points)
    angles = np.zeros(n)

    for i, p in enumerate(points):
        # Get closest neighbors
        distances = np.linalg.norm(points - p, axis=1)
        nearest_idx = np.argsort(distances)[1:5]  # 4 nearest neighbors
        neighbors = points[nearest_idx]

        if len(neighbors) >= 2:
            mean = np.mean(neighbors, axis=0)
            centered = neighbors - mean
            cov_matrix = np.dot(centered.T, centered)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            dominant_direction = eigenvectors[:, np.argmax(eigenvalues)]
            angles[i] = np.arctan2(dominant_direction[1], dominant_direction[0])

    return angles

def compute_shape_context(points, angles, nbins_r=5, nbins_theta=12, r_inner=0.125, r_outer=2.0):
    """Compute rotation-invariant Shape Context descriptors."""
    n = len(points)
    shape_contexts = np.zeros((n, nbins_r * nbins_theta))
    
    r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)
    theta_bin_edges = np.linspace(-np.pi, np.pi, nbins_theta, endpoint=False)

    for i, p in enumerate(points):
        distances = np.linalg.norm(points - p, axis=1)
        angles_rel = np.arctan2(points[:, 1] - p[1], points[:, 0] - p[0]) - angles[i]  # Normalize angle

        hist = np.zeros((nbins_r, nbins_theta))
        for d, a in zip(distances, angles_rel):
            if d == 0:
                continue
            r_bin = np.searchsorted(r_bin_edges, d, side='right')
            theta_bin = np.searchsorted(theta_bin_edges, a, side='right') % nbins_theta
            if r_bin < nbins_r:
                hist[r_bin, theta_bin] += 1

        shape_contexts[i, :] = hist.flatten()

    return shape_contexts

def match_shapes(descriptors1, descriptors2):
    """Match shapes using the Hungarian Algorithm based on Shape Context similarity."""
    if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
        return None, None

    cost_matrix = np.zeros((descriptors1.shape[0], descriptors2.shape[0]))
    for i in range(descriptors1.shape[0]):
        for j in range(descriptors2.shape[0]):
            h1, h2 = descriptors1[i], descriptors2[j]
            cost_matrix[i, j] = 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def draw_matches(img1, points1, img2, points2, matches):
    # Ensure both images have the same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 != h2:
        new_h = max(h1, h2)  # Find the max height
        img1 = cv2.resize(img1, (w1, new_h))  # Resize img1 to match height
        img2 = cv2.resize(img2, (w2, new_h))  # Resize img2 to match height

    match_img = np.hstack((img1, img2))  # Now images have the same height

    # Offset points from img2 since it's concatenated
    offset_x = img1.shape[1]
    
    for p1, p2 in matches:
        pt1 = tuple(map(int, points1[p1].ravel()))
        pt2 = tuple(map(int, points2[p2].ravel()))
        pt2 = (pt2[0] + offset_x, pt2[1])  # Shift x-coordinates

        cv2.line(match_img, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(match_img, pt1, 3, (255, 0, 0), -1)
        cv2.circle(match_img, pt2, 3, (0, 0, 255), -1)

    return match_img


# Load images
img1 = cv2.imread("bestref.png")
img2 = cv2.imread("target_shape.png")

# Detect corners
points1, gray1 = detect_corners(img1)
points2, gray2 = detect_corners(img2)

# Compute tangent angles (rotation-invariant)
angles1 = compute_tangent_angles(points1)
angles2 = compute_tangent_angles(points2)

# Compute rotation-invariant Shape Context descriptors
descriptors1 = compute_shape_context(points1, angles1)
descriptors2 = compute_shape_context(points2, angles2)

# Match shapes
matches = match_shapes(descriptors1, descriptors2)

# Draw results
if matches[0] is not None:
    matched_img = draw_matches(img1, points1, img2, points2, matches)
    cv2.imshow("Rotation-Invariant Shape Context Matching", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matches found.")
