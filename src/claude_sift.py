import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.measure import label, regionprops

def detect_object_with_sift(reference_img_path, input_img_path, min_kernel_size=64, max_kernel_size=512, min_matches=4):
    """
    Detect objects in an input image using SIFT matching with sliding window approach.
    
    Args:
        reference_img_path: Path to the reference object image
        input_img_path: Path to the input image where objects need to be detected
        min_kernel_size: Minimum size of the sliding window kernel
        max_kernel_size: Maximum size of the sliding window kernel
        min_matches: Minimum number of matches required to consider a detection
        
    Returns:
        Tuple containing (input_image with detections, best_match_region, best_match_count)
    """
    # Load images
    reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
    input_img = cv2.imread(input_img_path)
    input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    input_height, input_width = input_img_gray.shape
    ref_height, ref_width = reference_img.shape
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Extract keypoints and descriptors from reference image
    ref_keypoints, ref_descriptors = sift.detectAndCompute(reference_img, None)
    
    # Determine kernel size based on input image size
    kernel_size = min(max(min_kernel_size, int(min(input_width, input_height) / 8)), max_kernel_size)
    kernel_size = 256
    print(f"Using kernel size: {kernel_size}x{kernel_size}")
    
    # Initialize match map (same size as input image)
    match_map = np.zeros((input_height, input_width), dtype=np.uint8)
    
    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Define step size for sliding window (using 1/4 of kernel size for efficiency)
    step_size = kernel_size // 4
    
    # Sliding window approach
    print("Performing sliding window SIFT matching...")
    for y in range(0, input_height - kernel_size, step_size):
        for x in range(0, input_width - kernel_size, step_size):
            # Extract window
            window = input_img_gray[y:y+kernel_size, x:x+kernel_size]
            
            # Extract keypoints and descriptors
            window_keypoints, window_descriptors = sift.detectAndCompute(window, None)
            
            # Skip if no keypoints are found
            if window_keypoints and window_descriptors is not None and len(window_keypoints) > 0:
                # Match descriptors
                matches = flann.knnMatch(ref_descriptors, window_descriptors, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                # If enough matches are found, mark the region
                if len(good_matches) >= min_matches:
                    # Mark the central point of the window
                    center_y, center_x = y + kernel_size//2, x + kernel_size//2
                    match_map[center_y, center_x] = 1
                    
                    # Optionally mark the whole window (useful for visualization)
                    match_map[y:y+kernel_size, x:x+kernel_size] = 1
    
    # Label connected regions using 8-connectivity
    print("Labeling connected regions...")
    labeled_map, num_regions = label(match_map, connectivity=2, return_num=True)
    
    # Find properties of labeled regions
    regions = regionprops(labeled_map)
    
    # Initialize visualization image
    vis_img = input_img.copy()
    
    # Prepare for finding the best match
    best_match_count = 0
    best_match_region = None
    best_match_bbox = None
    
    # Process each region
    print(f"Processing {len(regions)} regions...")
    region_info = []
    
    for region in regions:
        # Get bounding box (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox
        
        # Calculate center of bounding box
        center_y = (min_row + max_row) // 2
        center_x = (min_col + max_col) // 2
        
        # Extract region around center with size of kernel
        start_y = max(0, center_y - kernel_size//2)
        start_x = max(0, center_x - kernel_size//2)
        end_y = min(input_height, center_y + kernel_size//2)
        end_x = min(input_width, center_x + kernel_size//2)
        
        region_img = input_img_gray[start_y:end_y, start_x:end_x]
        
        # Extract keypoints and descriptors
        region_keypoints, region_descriptors = sift.detectAndCompute(region_img, None)
        
        # Skip if no keypoints are found
        match_count = 0
        if region_keypoints and region_descriptors is not None and len(region_keypoints) > 0:
            # Match descriptors
            matches = flann.knnMatch(ref_descriptors, region_descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            match_count = len(good_matches)
        
        # Store region information
        region_bbox = (min_col, min_row, max_col - min_col, max_row - min_row)  # x, y, width, height
        region_info.append((region_bbox, match_count))
        
        # Update best match if this region has more matches
        if match_count > best_match_count:
            best_match_count = match_count
            best_match_region = region_img
            best_match_bbox = region_bbox
    
    # Draw bounding boxes with match counts on the visualization image
    for bbox, count in region_info:
        x, y, width, height = bbox
        color = (0, 255, 0) if (bbox == best_match_bbox) else (0, 0, 255)
        thickness = 3 if (bbox == best_match_bbox) else 2
        
        # Draw rectangle
        cv2.rectangle(vis_img, (x, y), (x + width, y + height), color, thickness)
        
        # Add text with match count
        cv2.putText(vis_img, f"Matches: {count}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_img, best_match_region, best_match_count

def visualize_results(input_img, reference_img, result_img, best_match_region, best_match_count):
    """
    Visualize the detection results with matplotlib.
    
    Args:
        input_img: Original input image
        reference_img: Reference object image
        result_img: Input image with bounding boxes
        best_match_region: The region with the best match
        best_match_count: Number of matches in the best region
    """
    plt.figure(figsize=(15, 10))
    
    # Plot reference image
    plt.subplot(221)
    plt.title('Reference Object')
    plt.imshow(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Plot input image
    plt.subplot(222)
    plt.title('Input Image')
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Plot result image with bounding boxes
    plt.subplot(223)
    plt.title('Detection Results')
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Plot best match region
    if best_match_region is not None:
        plt.subplot(224)
        plt.title(f'Best Match Region\n{best_match_count} matches')
        plt.imshow(cv2.cvtColor(best_match_region, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_results.png')
    plt.show()

def main():
    # Replace these with your actual image paths
    reference_path = 'To Embed/Snippet_Ref_2.jpg'
    input_path = 'To Embed/624_Marked.jpg'
    
    # Run detection
    print("Starting object detection...")
    result_img, best_match_region, best_match_count = detect_object_with_sift(
        reference_path, input_path, min_kernel_size=64, max_kernel_size=512, min_matches=4
    )
    
    # Display results
    print(f"Best match found with {best_match_count} matching points")
    
    # Load original images for visualization
    input_img = cv2.imread(input_path)
    reference_img = cv2.imread(reference_path)
    
    # Visualize results
    visualize_results(input_img, reference_img, result_img, best_match_region, best_match_count)
    
    # Save the result image
    cv2.imwrite('detection_result.jpg', result_img)
    print("Results saved as 'detection_result.jpg' and 'detection_results.png'")

if __name__ == "__main__":
    main()