import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def compute_kernel_size(image_shape):
    h, w = image_shape[:2]
    kernel_size = min(max(min(h, w) // 2, 64), 512)
    return kernel_size

def sift_match(des1, kp1, img_patch, sift, bf):
    kp2, des2 = sift.detectAndCompute(img_patch, None)
    if des2 is None or len(des2) < 2:
        return 0, []
    
    matches = bf.knnMatch(des1, des2, k=2)

    # Ensure each match has at least two elements
    good_matches = [m for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

    return len(good_matches), good_matches

def sliding_window_matching(input_image, reference_image):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kp1, des1 = sift.detectAndCompute(reference_image, None)
    kernel_size = compute_kernel_size(input_image.shape)
    h, w = input_image.shape[:2]
    match_map = np.zeros((h, w), dtype=np.uint8)
    step_control = kernel_size
    for y in range(0, h - kernel_size, 1):
      for x in range(0, w - kernel_size, 1):
          patch = input_image[y:y+kernel_size, x:x+kernel_size]
          num_matches, good_matches = sift_match(des1, kp1, patch, sift, bf)

          if num_matches >= 4:
              for match in good_matches:
                  kp = match[0].trainIdx  # Get matched keypoint index
                  keypoint_location = sift.detect(patch)[kp].pt  # Get coordinates in patch
                  
                  # Convert local patch keypoint to global image coordinates
                  key_x = int(x + keypoint_location[0])
                  key_y = int(y + keypoint_location[1])

                  # Ensure we don't exceed image boundaries
                  if 0 <= key_x < w and 0 <= key_y < h:
                      match_map[key_y, key_x] = 1  # Set matched point to 1

    
    return match_map, kernel_size

def draw_bounding_boxes(input_image, match_map, kernel_size, reference_image):
    labeled_array, num_features = label(match_map)
    best_match = None
    max_matches = 0
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kp1, des1 = sift.detectAndCompute(reference_image, None)
    
    img_display = input_image.copy()
    
    for region in range(1, num_features + 1):
        y_coords, x_coords = np.where(labeled_array == region)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        y1, y2 = max(cy - kernel_size//2, 0), min(cy + kernel_size//2, input_image.shape[0])
        x1, x2 = max(cx - kernel_size//2, 0), min(cx + kernel_size//2, input_image.shape[1])
        subimage = input_image[y1:y2, x1:x2]
        num_matches, _ = sift_match(des1, kp1, subimage, sift, bf)
        cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img_display, f"Matches: {num_matches}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if num_matches > max_matches:
            max_matches = num_matches
            best_match = (x1, y1, x2, y2)
    
    if best_match:
        cv2.rectangle(img_display, (best_match[0], best_match[1]), (best_match[2], best_match[3]), (0, 0, 255), 3)
        cv2.putText(img_display, "Best Match", (best_match[0], best_match[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return img_display

# Example Usage

input_image = cv2.imread("To Embed/Snippet_Ref_2.jpg", cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread("To Embed/Snippet_Ref_2.jpg", cv2.IMREAD_GRAYSCALE)
match_map, kernel_size = sliding_window_matching(input_image, reference_image)
result_image = draw_bounding_boxes(input_image, match_map, kernel_size, reference_image)

plt.figure(figsize=(10, 6))
plt.imshow(result_image, cmap='gray')
plt.title("Detected Regions with Bounding Boxes")
plt.axis("off")
plt.show()
