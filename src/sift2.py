import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_sift_features(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image with keypoints
    plt.figure(figsize=(10, 6))
    plt.imshow(image_with_keypoints, cmap='gray')
    plt.title(f"SIFT Keypoints ({len(keypoints)} detected)")
    plt.axis("off")
    plt.show()

# Example usage
visualize_sift_features("Reference_Taxi.jpg")
