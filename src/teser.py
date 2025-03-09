import cv2
import numpy as np

# Load images
source = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('reference.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if source is None or template is None:
    raise ValueError("Error loading images. Check file paths.")

# Apply Canny edge detection
edges1 = cv2.Canny(source, 50, 150)
edges2 = cv2.Canny(template, 50, 150)

# Find contours
contours_src, _ = cv2.findContours(edges1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_tpl, _ = cv2.findContours(edges2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Check if contours are found
if not contours_src or not contours_tpl:
    raise ValueError("No contours found in one or both images.")

# Convert grayscale images to BGR for visualization
source_vis = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
template_vis = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)

# Store best matches (each contour in source finds its best match in template)
best_matches = []

for src_contour in contours_src:
    best_match = None
    best_score = float('inf')

    for tpl_contour in contours_tpl:
        score = cv2.matchShapes(src_contour, tpl_contour, 1, 0.0)
        if score < best_score:
            best_score = score
            best_match = tpl_contour
            print(best_score)

    best_matches.append((src_contour, best_match, best_score))

# Sort best matches by similarity score (lower is better)
best_matches.sort(key=lambda x: x[2])

# Draw the best matches
for i, (src_contour, tpl_contour, score) in enumerate(best_matches[:10]):  # Show top 10 matches
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # Random color
    cv2.drawContours(source_vis, [src_contour], -1, color, 2)
    cv2.drawContours(template_vis, [tpl_contour], -1, color, 2)

# Resize template to match source height
h1, w1 = source_vis.shape[:2]
h2, w2 = template_vis.shape[:2]
new_width = w2 * h1 // h2  # Preserve aspect ratio
template_vis = cv2.resize(template_vis, (new_width, h1))

# Stack images side by side
combined = np.hstack((source_vis, template_vis))

# Show the matched contours
cv2.imshow("Matched Contours", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
