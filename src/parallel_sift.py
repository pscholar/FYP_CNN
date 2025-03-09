import cv2
import numpy as np
import concurrent.futures

def sift_match(des1, kp1, img_patch, sift, bf):
    """Performs SIFT matching between reference descriptors and an image patch."""
    kp2, des2 = sift.detectAndCompute(img_patch, None)
    if des2 is None or len(des2) < 2:
        return 0, []
    
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

    return len(good_matches), good_matches

def process_strip(y_range, input_image, reference_image, kernel_size, sift, bf, des1, kp1):
    """Processes a horizontal strip of the image to speed up matching."""
    h, w = input_image.shape[:2]
    local_match_map = np.zeros((h, w), dtype=np.uint8)

    for y in y_range:
        for x in range(0, w - kernel_size + 1, 1):  
            # Handle edge conditions
            patch = input_image[y:min(y+kernel_size, h), x:min(x+kernel_size, w)]
            
            num_matches, _ = sift_match(des1, kp1, patch, sift, bf)
            if num_matches > 4:
                local_match_map[y:y+patch.shape[0], x:x+patch.shape[1]] = 1  # Only within bounds

    return local_match_map

def compute_kernel_size(image_shape):
    """Dynamically computes the kernel size based on image dimensions."""
    h, w = image_shape[:2]
    return min(max(64, min(h, w) // 8), 512)  # Between 64x64 and 512x512

def sliding_window_matching(input_image, reference_image, num_threads=4):
    """Runs SIFT-based sliding window matching with parallel processing."""
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kp1, des1 = sift.detectAndCompute(reference_image, None)
    kernel_size = compute_kernel_size(input_image.shape)
    h, w = input_image.shape[:2]
    match_map = np.zeros((h, w), dtype=np.uint8)

    # Define thread workload by splitting image into horizontal strips
    step = max(1, h // num_threads)  
    y_ranges = [range(i, min(i + step, h - kernel_size + 1), 1) for i in range(0, h - kernel_size + 1, step)]
    
    # Parallel processing of image strips
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(process_strip, y_ranges, 
                               [input_image] * len(y_ranges), [reference_image] * len(y_ranges), 
                               [kernel_size] * len(y_ranges), [sift] * len(y_ranges), 
                               [bf] * len(y_ranges), [des1] * len(y_ranges), [kp1] * len(y_ranges))

    # Combine results from all threads
    for res in results:
        match_map = np.bitwise_or(match_map, res)  

    return match_map, kernel_size

def draw_bounding_boxes(image, match_map, kernel_size):
    """Finds connected regions in match_map and draws bounding boxes."""
    contours, _ = cv2.findContours(match_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output_image, f"{w}x{h}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output_image

def main():
    """Main function to run SIFT-based object detection with visualization."""
    # Load images
    reference_image = cv2.imread("To Embed/Snippet_Ref_3.jpg", cv2.IMREAD_GRAYSCALE)
    input_image = cv2.imread("To Embed/Reference_Taxi_Body_Outline.jpg", cv2.IMREAD_GRAYSCALE)
    
    if reference_image is None or input_image is None:
        print("Error: Could not load images.")
        return

    # Perform sliding window SIFT matching
    match_map, kernel_size = sliding_window_matching(input_image, reference_image)

    # Draw bounding boxes
    result_image = draw_bounding_boxes(cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR), match_map, kernel_size)

    # Display results
    cv2.imshow("Match Map", match_map * 255)  # Convert binary map to displayable format
    cv2.imshow("Detected Regions", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
