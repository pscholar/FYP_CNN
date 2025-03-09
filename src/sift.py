import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_sift(image_path):
    """
    Reads an image, detects SIFT keypoints, and visualizes them.
    
    :param image_path: Path to the input image
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not read the image.")
        return
    
    # Create a SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show the image with keypoints
    plt.figure(figsize=(10, 6))
    plt.imshow(image_with_keypoints, cmap='gray')
    plt.title(f"SIFT Keypoints ({len(keypoints)} detected)")
    plt.axis('off')
    plt.show()

# Example usage
# visualize_sift("path_to_your_image.jpg")

def select_roi(image, window_name="Select ROI"):
    """Allows manual polygon selection on a resized image, then scales ROI back to original size."""
    h_original, w_original = image.shape[:2]
    scale_factor = min(800 / h_original, 1200 / w_original)  # Scale to fit within 1200x800
    resized = cv2.resize(image, (int(w_original * scale_factor), int(h_original * scale_factor)))

    mask = np.zeros(resized.shape[:2], dtype=np.uint8)
    points = []

    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add points
            points.append((x, y))
            cv2.circle(temp_img, (x, y), 3, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(temp_img, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow(window_name, temp_img)

        elif event == cv2.EVENT_RBUTTONDOWN and len(points) > 2:  # Right-click to finalize
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
            cv2.destroyAllWindows()

    temp_img = resized.copy()
    cv2.imshow(window_name, temp_img)
    cv2.setMouseCallback(window_name, draw_polygon)
    cv2.waitKey(0)

    # Resize the mask back to original size
    mask_full_res = cv2.resize(mask, (w_original, h_original), interpolation=cv2.INTER_NEAREST)

    return mask_full_res


# Preprocessing function
def preprocess_image(image, mask):
    """Applies CLAHE and Gaussian blur only to the masked region."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return cv2.bitwise_and(gray, gray, mask=mask)  # Keep only the selected region


# Main function
def register_images_siftv5(reference_img_path, target_img_path):
    """Registers images by selecting ROIs for feature matching and manually selecting a blending region."""
    # Load images
    ref_img = cv2.imread(reference_img_path, cv2.IMREAD_COLOR)
    target_img = cv2.imread(target_img_path, cv2.IMREAD_COLOR)

    # Manually select ROIs
    print("Select ROI on the Reference Image")
    ref_mask = select_roi(ref_img)

    print("Select ROI on the Target Image")
    target_mask = select_roi(target_img)

    # Preprocess images (apply ROI mask)
    ref_gray = preprocess_image(ref_img, ref_mask)
    target_gray = preprocess_image(target_img, target_mask)

    # Detect features in the selected region using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    # Visualize keypoints
    img_kp1 = cv2.drawKeypoints(ref_gray, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_kp2 = cv2.drawKeypoints(target_gray, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_kp1, cmap="gray")
    plt.title("SIFT Keypoints - Reference Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_kp2, cmap="gray")
    plt.title("SIFT Keypoints - Target Image")
    plt.axis("off")
    plt.show()
    # Feature matching using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Ensure sufficient matches
    if len(good_matches) < 4:
        print("Not enough good matches found!")
        return None

    # Extract matched keypoints
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography transformation
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    # Warp the entire target image
    h, w, _ = ref_img.shape
    warped_img = cv2.warpPerspective(target_img, H, (w, h))

    # Manually select the blending region
    print("Select the region from the Warped Image for Blending")
    blend_mask = select_roi(warped_img)

    # Blend selected region
    blended = cv2.bitwise_and(ref_img,ref_img,mask=ref_mask)

        #ref_mask.copy()
    blend_region = cv2.bitwise_and(warped_img, warped_img, mask=blend_mask)
    blended = cv2.addWeighted(blended, 0.5, blend_region, 0.5, 0)

    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Draw matches
    match_img = cv2.drawMatches(ref_img, kp1, target_img, kp2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[0].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Feature Matches (SIFT + BFMatcher)")
    axes[0].axis("off")

    # Show warped image
    axes[1].imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Warped Target Image")
    axes[1].axis("off")

    # Show final overlaid result
    axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Overlay: Reference + Selected Warped Region")
    axes[2].axis("off")

    plt.show()

    return blended


# Example usage
REFERENCE_IMG_PATH = "Reference_Taxi"
TARGET_IMG_PATH = "input.jpg"
register_images_siftv5(REFERENCE_IMG_PATH, TARGET_IMG_PATH)
#register_images_orb(REFERENCE_IMG_PATH, TARGET_IMG_PATH)

