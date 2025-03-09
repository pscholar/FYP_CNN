import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_sift_features(image1, image2, num_matches=100,
                        nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                        edgeThreshold=10, sigma=1.6, ratio_thresh=0.5):
    """Matches SIFT features between two images using BFMatcher with Lowe's ratio test and returns keypoints and good matches."""
    if image1 is None or image2 is None:
        raise ValueError("Error: One or both input images are None.")
    
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers,
                           contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold,
                           sigma=sigma)
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    bf = cv2.BFMatcher()  # Brute Force Matcher without cross-checking
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # Find the two nearest neighbors for each descriptor  
    good_matches = []
    
    # Apply Lowe's ratio test
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    # Sort the matches based on the distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:num_matches]
    
    return keypoints1, keypoints2, good_matches

def find_best_reference(ref1, ref2, target):
    """Compares two reference images against a target image and selects the best match."""
    keypoints1, keypoints_target1, good_matches1 = match_sift_features(ref1, target)
    keypoints2, keypoints_target2, good_matches2 = match_sift_features(ref2, target)
    
    if len(good_matches1) > len(good_matches2):
        return ref1, keypoints1, keypoints_target1, good_matches1, 1  # Unflipped is better
    else:
        return ref2, keypoints2, keypoints_target2, good_matches2, -1  # Flipped is better

def visualize_matches(ref_image, target_image, keypoints1, keypoints2, good_matches):
    """Visualizes the best-matched reference image with the target image."""
    match_img = cv2.drawMatches(ref_image, keypoints1, target_image, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("sift_matched_points.jpg", match_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    plt.figure()
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT Matches ({len(good_matches)} good matches)")
    plt.axis("off")
    plt.show()

def compute_homography(best_ref, keypoints1, keypoints2, good_matches):
    if len(good_matches) < 4:
        print("Not enough matches to compute homography.")
        return None
    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # Compute the homography matrix
    H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 5.0)
    return H, mask

if __name__ == "__main__":
# Example usage
  ref1 = cv2.imread("To Embed/Reference_Taxi_Body_Outline.jpg")
  ref2 = cv2.imread("To Embed/Reference_Taxi_Body_Outline_Flipped.jpg")
  target = cv2.imread("To Embed/to12_Marked.jpg")
  best_ref, keypoints1, keypoints2, good_matches, flag = find_best_reference(ref1, ref2, target)
  if flag == 1:
      print("Use unflipped Image")
  else:
      print("Use Flipped Image")
  visualize_matches(best_ref, target, keypoints1, keypoints2, good_matches)
  H, mask = compute_homography(best_ref, keypoints1, keypoints2, good_matches)
  print(f"Homography: {H}",)
