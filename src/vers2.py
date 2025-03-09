import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_flood_taxi_detection(image, results):
    """
    Process Mask R-CNN detection results to identify taxi-flood overlaps
    and generate appropriate masks and images using HSV color space.
    
    Args:
        image: Original input image (numpy array with shape [H, W, 3] in BGR format)
        results: Output from model.detect([image], verbose=0)
                 Contains 'rois', 'masks', 'class_ids', and 'scores'
    
    Returns:
        processed_results: Dictionary containing:
            - 'flood_masks': List of binary masks for flood regions in ROIs
            - 'taxi_regions': List of images containing taxi pixels
            - 'overlap_indices': List of (taxi_idx, flood_idx) pairs that overlap
            - 'visualization': Visualization of the final results
    """
    # Extract results
    rois = results[0]['rois']  # [N, (y1, x1, y2, x2)] - Mask RCNN stores as y,x format
    masks = results[0]['masks']  # [H, W, N]
    class_ids = results[0]['class_ids']  # [N]
    scores = results[0]['scores']  # [N]
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    # Convert image to HSV for better color analysis
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Convert masks to the format we need
    masks = np.transpose(masks, (2, 0, 1))  # [N, H, W]
    
    # Separate taxi and flood detections
    taxi_indices = np.where(class_ids == 2)[0]  # Assuming 'taxi' is class_id 2
    flood_indices = np.where(class_ids == 1)[0]  # Assuming 'flood' is class_id 1
    
    taxi_boxes = rois[taxi_indices]  # [y1, x1, y2, x2]
    flood_boxes = rois[flood_indices]  # [y1, x1, y2, x2]
    
    taxi_masks = masks[taxi_indices]
    flood_masks = masks[flood_indices]
    
    # Initialize result containers
    processed_results = {
        'flood_masks': [],
        'taxi_regions': [],
        'overlap_indices': [],
        'visualizations': []
    }
    
    # Create visualization image (make a deep copy to avoid modifying the original)
    vis_image = image.copy()
    
    # For each taxi, check for overlaps with floods
    for i, taxi_box in enumerate(taxi_boxes):
        taxi_mask = taxi_masks[i]
        T_y1, T_x1, T_y2, T_x2 = taxi_box  # Unpacking in y,x format
        
        for j, flood_box in enumerate(flood_boxes):
            flood_mask = flood_masks[j]
            F_y1, F_x1, F_y2, F_x2 = flood_box  # Unpacking in y,x format
            
            # Check if taxi overlaps with flood vertically
            overlaps = (T_y2 <= F_y2 and 
                       ((T_x1 >= F_x1 and T_x1 <= F_x2) or 
                        (T_x2 >= F_x1 and T_x2 <= F_x2)))
            
            if overlaps:
                processed_results['overlap_indices'].append((taxi_indices[i], flood_indices[j]))
                
                # Find the lowest points of the taxi mask
                taxi_mask_points = np.where(taxi_mask)
                y_coords = taxi_mask_points[0]
                x_coords = taxi_mask_points[1]
                
                # Group points by their x coordinate
                x_to_y = {}
                for idx in range(len(x_coords)):
                    x = x_coords[idx]
                    y = y_coords[idx]
                    if x not in x_to_y:
                        x_to_y[x] = []
                    x_to_y[x].append(y)
                
                # Find the lowest point for each x coordinate
                lowest_points = {}
                for x, ys in x_to_y.items():
                    lowest_points[x] = max(ys)
                
                # Find the leftmost and rightmost coordinates of the taxi mask
                leftmost_x = min(x_coords) if len(x_coords) > 0 else T_x1
                rightmost_x = max(x_coords) if len(x_coords) > 0 else T_x2
                
                # Get their corresponding y-coordinates (lowest points)
                leftmost_y = lowest_points.get(leftmost_x, T_y2)
                rightmost_y = lowest_points.get(rightmost_x, T_y2)
                
                # Define the ROI
                roi_bottom_y = int(T_y2 + (F_y2 - F_y1) / 10)
                
                # Create a polygon representing the ROI
                roi_polygon = np.array([
                    [T_x1, leftmost_y],
                    [T_x2, rightmost_y],
                    [T_x2, roi_bottom_y],
                    [T_x1, roi_bottom_y]
                ], dtype=np.int32)
                
                # Create a mask for the ROI
                h, w = image.shape[:2]
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(roi_mask, [roi_polygon], 1)
                
                # Identify flood pixels within the ROI
                flood_roi = np.logical_and(flood_mask, roi_mask).astype(np.uint8)
                
                # If there are flood pixels in the ROI
                if np.sum(flood_roi) > 0:
                    # IMPROVED COLOR THRESHOLDING USING HSV
                    flood_hsv_pixels = []
                    flood_y_coords, flood_x_coords = np.where(flood_roi == 1)
                    
                    for idx in range(len(flood_y_coords)):
                        y, x = flood_y_coords[idx], flood_x_coords[idx]
                        flood_hsv_pixels.append(image_hsv[y, x])
                    
                    if len(flood_hsv_pixels) > 10:
                        flood_hsv_pixels = np.array(flood_hsv_pixels)
                        
                        # Calculate mean and standard deviation in HSV space
                        hsv_mean = np.mean(flood_hsv_pixels, axis=0)
                        hsv_std = np.std(flood_hsv_pixels, axis=0)
                        
                        # Handle hue circular nature - it's an angle, so we need special handling
                        # Hue values in OpenCV are in [0, 180) range
                        hue_values = flood_hsv_pixels[:, 0]
                        
                        # Special handling for hue - use circular statistics
                        sin_sum = np.sum(np.sin(hue_values * np.pi / 90))
                        cos_sum = np.sum(np.cos(hue_values * np.pi / 90))
                        mean_hue = np.arctan2(sin_sum, cos_sum) * 90 / np.pi
                        if mean_hue < 0:
                            mean_hue += 180
                            
                        # Replace calculated mean with circular mean for hue
                        hsv_mean[0] = mean_hue
                        
                        # Create HSV thresholds - being more strict with hue,
                        # but more lenient with saturation and value
                        h_lower = max(0, hsv_mean[0] - 15)
                        h_upper = min(180, hsv_mean[0] + 15)
                        
                        # For saturation and value, use larger thresholds
                        s_lower = max(0, hsv_mean[1] - 50)
                        s_upper = min(255, hsv_mean[1] + 50)
                        
                        v_lower = max(0, hsv_mean[2] - 50)
                        v_upper = min(255, hsv_mean[2] + 50)
                        
                        # Create a mask for pixels within these HSV thresholds
                        lower_bound = np.array([h_lower, s_lower, v_lower])
                        upper_bound = np.array([h_upper, s_upper, v_upper])
                        
                        # Special handling for hue wrapping around (e.g., if range crosses 180/0 boundary)
                        if h_upper < h_lower:  # Hue wraps around
                            # Create two masks and combine them
                            lower_mask1 = np.array([0, s_lower, v_lower])
                            upper_mask1 = np.array([h_upper, s_upper, v_upper])
                            
                            lower_mask2 = np.array([h_lower, s_lower, v_lower])
                            upper_mask2 = np.array([180, s_upper, v_upper])
                            
                            hsv_mask1 = cv2.inRange(image_hsv, lower_mask1, upper_mask1)
                            hsv_mask2 = cv2.inRange(image_hsv, lower_mask2, upper_mask2)
                            hsv_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)
                        else:
                            hsv_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
                        
                        # Apply ROI constraint
                        refined_flood_mask = np.zeros((h, w), dtype=np.uint8)
                        refined_flood_mask[np.logical_and(roi_mask == 1, hsv_mask > 0)] = 1
                        
                        # Optional: Apply morphological operations to clean up the mask
                        kernel = np.ones((3, 3), np.uint8)
                        refined_flood_mask = cv2.morphologyEx(refined_flood_mask, cv2.MORPH_OPEN, kernel)
                        refined_flood_mask = cv2.morphologyEx(refined_flood_mask, cv2.MORPH_CLOSE, kernel)
                    else:
                        # Fallback if not enough pixels
                        refined_flood_mask = flood_roi
                    
                    # Create taxi region image
                    taxi_region = np.zeros_like(image)
                    taxi_y_coords, taxi_x_coords = np.where(taxi_mask)
                    for idx in range(len(taxi_x_coords)):
                        x, y = taxi_x_coords[idx], taxi_y_coords[idx]
                        if T_x1 <= x <= T_x2 and T_y1 <= y <= T_y2:
                            taxi_region[y, x] = image[y, x]
                    
                    # Create visualization - using BGR format for OpenCV display
                    visualization = image.copy()
                    
                    # Draw taxi bounding box in blue
                    cv2.rectangle(visualization, (T_x1, T_y1), (T_x2, T_y2), (255, 0, 0), 2)
                    
                    # Draw ROI polygon in yellow
                    cv2.polylines(visualization, [roi_polygon], True, (0, 255, 255), 2)
                    
                    # Add flood mask overlay in red with transparency
                    flood_overlay = visualization.copy()
                    flood_overlay[refined_flood_mask == 1] = [0, 0, 255]  # BGR color order for OpenCV
                    alpha = 0.5
                    beta = 1.0 - alpha
                    cv2.addWeighted(flood_overlay, alpha, visualization, beta, 0, visualization)
                    
                    # Add taxi mask overlay in green with transparency
                    taxi_overlay = visualization.copy()
                    taxi_y_coords, taxi_x_coords = np.where(taxi_mask)
                    for idx in range(len(taxi_x_coords)):
                        x, y = taxi_x_coords[idx], taxi_y_coords[idx]
                        taxi_overlay[y, x] = [0, 255, 0]  # BGR color order for OpenCV
                    alpha = 0.3
                    beta = 1.0 - alpha
                    cv2.addWeighted(taxi_overlay, alpha, visualization, beta, 0, visualization)
                    
                    # Add text annotations
                    cv2.putText(visualization, "Taxi", (T_x1, T_y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(visualization, f"Flood (overlap {j+1})", (F_x1, F_y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Add results to the output
                    processed_results['flood_masks'].append(refined_flood_mask)
                    processed_results['taxi_regions'].append(taxi_region)
                    processed_results['visualizations'].append(visualization)
                    
                    # Update the main visualization
                    vis_image = visualization
    
    # Add the main visualization
    if len(processed_results['visualizations']) == 0 and (len(taxi_indices) > 0 or len(flood_indices) > 0):
        # Draw all detections if no overlaps found
        for i, taxi_box in enumerate(taxi_boxes):
            T_y1, T_x1, T_y2, T_x2 = taxi_box
            cv2.rectangle(vis_image, (T_x1, T_y1), (T_x2, T_y2), (255, 0, 0), 2)
            cv2.putText(vis_image, "Taxi", (T_x1, T_y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        for j, flood_box in enumerate(flood_boxes):
            F_y1, F_x1, F_y2, F_x2 = flood_box
            cv2.rectangle(vis_image, (F_x1, F_y1), (F_x2, F_y2), (0, 0, 255), 2)
            cv2.putText(vis_image, "Flood", (F_x1, F_y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    processed_results['main_visualization'] = vis_image
    
    return processed_results

def visualize_results(processed_results, figsize=(15, 10)):
    """
    Visualize the processed results with proper color conversion for display.
    
    Args:
        processed_results: Output from process_flood_taxi_detection function
        figsize: Size of the figure for matplotlib
    """
    visualizations = processed_results.get('visualizations', [])
    main_vis = processed_results.get('main_visualization')
    
    if main_vis is not None:
        plt.figure(figsize=figsize)
        # Convert BGR to RGB for matplotlib display
        plt.imshow(cv2.cvtColor(main_vis, cv2.COLOR_BGR2RGB))
        plt.title("Taxi and Flood Detection with Overlaps")
        plt.axis('off')
        plt.show()
    
    if len(visualizations) > 0:
        n_vis = min(4, len(visualizations))
        rows = (n_vis + 1) // 2
        cols = min(2, n_vis)
        
        plt.figure(figsize=figsize)
        for i, vis in enumerate(visualizations[:n_vis]):
            plt.subplot(rows, cols, i+1)
            # Convert BGR to RGB for matplotlib display
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Overlap {i+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # If there are more than 4 visualizations, show them in batches
    if len(visualizations) > 4:
        for batch_idx in range(1, (len(visualizations) + 3) // 4):
            start = batch_idx * 4
            end = min(start + 4, len(visualizations))
            if start < end:
                n_vis = end - start
                rows = (n_vis + 1) // 2
                cols = min(2, n_vis)
                
                plt.figure(figsize=figsize)
                for i in range(start, end):
                    plt.subplot(rows, cols, (i - start) + 1)
                    # Convert BGR to RGB for matplotlib display
                    plt.imshow(cv2.cvtColor(visualizations[i], cv2.COLOR_BGR2RGB))
                    plt.title(f"Overlap {i+1}")
                    plt.axis('off')
                plt.tight_layout()
                plt.show()

def process_results_with_external_function(image, detection_results, external_processing_function):
    """
    Process detection results and call an external function with appropriate inputs.
    
    Args:
        image: Original input image
        detection_results: Output from model.detect([image], verbose=0)
        external_processing_function: Function to call with processed results
        
    Returns:
        Results from the external processing function
    """
    # Process the detection results
    processed_data = process_flood_taxi_detection(image, detection_results)
    
    # Call the external function for each overlapping pair
    results = []
    for i in range(len(processed_data['overlap_indices'])):
        flood_mask = processed_data['flood_masks'][i]
        taxi_region = processed_data['taxi_regions'][i]
        result = external_processing_function(flood_mask, taxi_region, image)
        results.append(result)
    
    # Visualize the results
    visualize_results(processed_data)
    
    return results

# Example usage
"""
# Load image and run the model
image = cv2.imread('your_image.jpg')  # OpenCV loads images in BGR format
results = model.detect([image], verbose=0)

# Process the results
processed_data = process_flood_taxi_detection(image, results)

# Visualize
visualize_results(processed_data)

# Or with an external processing function
def your_processing_function(flood_mask, taxi_region, original_image):
    # Your custom processing logic
    return processed_result

final_results = process_results_with_external_function(image, results, your_processing_function)
"""