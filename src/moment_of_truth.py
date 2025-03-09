import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables to store selected points
points1, points2 = [], []
image1, image2 = None, None

def select_points(event, x, y, flags, param):
    """ Mouse callback function to store selected points """
    global points1, points2, selecting_first_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if selecting_first_image:
            points1.append((x, y))
            cv2.circle(image1_display, (x, y), 5, (0, 0, 255), -1)  # Red dot
            cv2.imshow("Select points in first image", image1_display)
        else:
            points2.append((x, y))
            cv2.circle(image2_display, (x, y), 5, (0, 0, 255), -1)  # Red dot
            cv2.imshow("Select points in second image", image2_display)

def get_homography(img1, img2):
    """ Allows user to select corresponding points and computes homography """
    global image1, image2, image1_display, image2_display, selecting_first_image
    image1, image2 = img1.copy(), img2.copy()
    image1_display, image2_display = img1.copy(), img2.copy()
    
    # Select points in the first image
    selecting_first_image = True
    cv2.imshow("Select points in first image", image1_display)
    cv2.setMouseCallback("Select points in first image", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Select points in the second image
    selecting_first_image = False
    cv2.imshow("Select points in second image", image2_display)
    cv2.setMouseCallback("Select points in second image", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ensure at least 4 points are selected
    if len(points1) < 4 or len(points2) < 4:
        print("Error: Select at least 4 points in both images!")
        return None, None

    # Convert points to NumPy array with correct shape
    pts1 = np.array(points1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.array(points2, dtype=np.float32).reshape(-1, 2)

    # Compute homography
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    print("Homography Matrix:\n", H)

    # Visualize matched points
    matched_img = draw_matches(image1, points1, image2, points2)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(matched_img,cv2.COLOR_BGR2RGB))
    plt.title("Matched Points")
    plt.tight_layout()
    plt.show()
    # cv2.imshow("Matched Points", matched_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Warp first image using homography
    h, w, _ = img2.shape
    warped_img = cv2.warpPerspective(img1, H, (w, h))

    # Overlay images
    overlay = cv2.addWeighted(img2, 0.5, warped_img, 0.5, 0)
    cv2.imshow("Warped Image Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H, matched_img


def draw_matches(img1, pts1, img2, pts2):
    """ Draws lines connecting corresponding points """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create a side-by-side image
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = img1
    result[:h2, w1:w1 + w2] = img2

    # Draw matching lines
    for p1, p2 in zip(pts1, pts2):
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]) + w1, int(p2[1]))  # Adjust for second image position
        cv2.line(result, pt1, pt2, (0, 255, 0), 2)  # Green line
        cv2.circle(result, pt1, 5, (0, 0, 255), -1)  # Red dot
        cv2.circle(result, pt2, 5, (0, 0, 255), -1)  # Red dot
    
    return result

# Load your images
img1 = cv2.imread("Reference_Taxi_Body_Outline.jpg")
img2 = cv2.imread("624_Marked.jpg")

# Run the function
H, matched_img = get_homography(img1, img2)
