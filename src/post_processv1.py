import numpy as np
import cv2

def draw_contour_line(binary_image, blend_image, line_color=(0, 255, 0), alpha=0.6):
    """
    Draws a line connecting the first nonzero pixel in each column of a binary 3-channel image.
    Blends this modified image with another image using alpha blending.

    Parameters:
        binary_image (numpy.ndarray): 3-channel binary image (shape: HxWx3).
        blend_image (numpy.ndarray): Image of the same size to blend with.
        line_color (tuple): Color of the line in (B, G, R) format.
        alpha (float): Blending factor (0 to 1).

    Returns:
        blended_image (numpy.ndarray): The final blended image.
    """
    # Convert to grayscale to find nonzero pixels
    gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    h, w, _ = binary_image.shape

    # Find first nonzero pixels in each column
    first_nonzero_points = []
    for col in range(w):
        nonzero_rows = np.where(gray[:, col] > 0)[0]
        if len(nonzero_rows) > 0:
            first_nonzero_points.append((col, nonzero_rows[0]))  # (x, y)

    # Copy the binary image for drawing
    drawn_image = binary_image.copy()

    # Draw the contour line
    for i in range(len(first_nonzero_points) - 1):
        cv2.line(drawn_image, first_nonzero_points[i], first_nonzero_points[i + 1], line_color, 2)

    # Apply alpha blending
    blended_image = cv2.addWeighted(drawn_image, alpha, blend_image, 1 - alpha, 0)

    return blended_image

dimport numpy as np
import cv2

def process_detection_results(model_results, image):
    """
    Processes Mask R-CNN detection results to analyze taxi-flood overlap,
    define a polygonal region of interest (ROI), filter flood pixels using adaptive HSV color similarity,
    and generate output images.

    Parameters:
        model_results (dict): The output of model.detect([image], verbose=0).
        image (numpy.ndarray): The original image.

    Returns:
        new_flood_mask (numpy.ndarray): Binary flood mask in the ROI.
        new_taxi_region (numpy.ndarray): Taxi region image.
    """
    # Extract detections
    rois = model_results[0]['rois']  # Bounding boxes
    masks = model_results[0]['masks']  # Segmentation masks
    class_ids = model_results[0]['class_ids']  # Class labels

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Ensure OpenCV format (BGR)

    # Ensure there are detections
    if len(rois) == 0:
        return None, None

    # Identify indices for taxi and flood
    taxis = [i for i, cls in enumerate(class_ids) if cls == 2]  # Assuming 'taxi' is class 2
    floods = [i for i, cls in enumerate(class_ids) if cls == 1]  # Assuming 'flood' is class 1

    # Initialize output masks
    new_flood_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)  # Binary mask
    new_taxi_region = np.zeros_like(image)  # Taxi region in color

    for t_idx in taxis:
        T_y1, T_x1, T_y2, T_x2 = rois[t_idx]
        taxi_mask = masks[:, :, t_idx]  # Taxi segmentation mask

        for f_idx in floods:
            F_y1, F_x1, F_y2, F_x2 = rois[f_idx]
            flood_mask = masks[:, :, f_idx]  # Flood segmentation mask

            # Check if taxi is within flood bounding box region
            if (
                (T_y2 <= F_y2 and T_x1 >= F_x1 and T_x1 <= F_x2) or
                (T_y2 <= F_y2 and T_x2 >= F_x1 and T_x2 <= F_x2)
            ):
                # Get bottom pixels of taxi mask
                bottom_pixels = np.argwhere(taxi_mask)
                if bottom_pixels.size == 0:
                    continue

                bottom_y_values = {}
                for y, x in bottom_pixels:
                    bottom_y_values[x] = max(bottom_y_values.get(x, y), y)

                sorted_x = sorted(bottom_y_values.keys())
                bottom_points = [(x, bottom_y_values[x]) for x in sorted_x]

                # Define polygon ROI with additional extension
                extension_y = int(T_y2 + (F_y2 - F_y1) / 4)
                roi_polygon = [
                    (T_x1, bottom_y_values[T_x1]),  # Leftmost bottom pixel
                    *bottom_points,  # All bottom pixels
                    (T_x2, bottom_y_values[T_x2]),  # Rightmost bottom pixel
                    (T_x2, extension_y),  # Extended bottom right
                    (T_x1, extension_y)  # Extended bottom left
                ]

                roi_polygon = np.array(roi_polygon, dtype=np.int32)
                
                # Create ROI mask (polygonal region)
                roi_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
                cv2.fillPoly(roi_mask, [roi_polygon], 1)  # Fill polygonal region

                # Exclude taxi mask from ROI
                roi_mask[taxi_mask == 1] = 0

                # Convert image to HSV
                smoothed_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # Apply Gaussian smoothing only in ROI
                smoothed_hsv_copy = smoothed_hsv.copy()
                for channel in range(3):  # Process each channel
                    channel_data = smoothed_hsv[:, :, channel]
                    smoothed_channel = cv2.GaussianBlur(channel_data, (3, 3), 0)
                    smoothed_hsv_copy[:, :, channel] = np.where(
                        roi_mask == 1,  # Apply smoothing only in ROI
                        smoothed_channel,
                        channel_data
                    )

                # Get flood pixels in ROI
                flood_points_in_roi = np.argwhere((roi_mask == 1) & (flood_mask == 1))

                if flood_points_in_roi.size > 0:
                    # Extract HSV values of flood pixels in the ROI
                    flood_hsv_values = np.array([
                        smoothed_hsv_copy[y, x] for y, x in flood_points_in_roi
                    ])

                    # Compute adaptive threshold using MAD
                    adaptive_threshold = adaptive_color_threshold(flood_hsv_values)

                    # Compute median flood color in HSV
                    median_flood_hsv = np.median(flood_hsv_values, axis=0)

                    # Iterate over all pixels in ROI
                    for y in range(image.shape[0]):
                        for x in range(image.shape[1]):
                            if roi_mask[y, x] == 1 and taxi_mask[y, x] == 0:
                                pixel_hsv = smoothed_hsv_copy[y, x]
                                color_distance = np.linalg.norm(pixel_hsv - median_flood_hsv)

                                # Apply thresholding
                                if color_distance < adaptive_threshold:
                                    new_flood_mask[y, x] = 255  # Mark as flood

                # Extract taxi region pixels
                new_taxi_region[T_y1:T_y2, T_x1:T_x2] = image[T_y1:T_y2, T_x1:T_x2] * taxi_mask[T_y1:T_y2, T_x1:T_x2, None]

    return new_flood_mask, new_taxi_region


    mask = cv2.merge([new_flood_mask] * 3)
    blend = draw_contour_line(mask,image)
    y11, x11, y21, x21 = rois[taxis[0]]
    cv2.rectangle(blend, (x11, y11), (x21, y21), (0, 0, 255), 2)
    y12, x12, y22, x22 = rois[floods[0]]
    cv2.rectangle(blend, (x12, y12), (x22, y22), (255, 0, 0), 2)
    bottom_y = y21 + (y22 - y12) // 4
    roi_x1, roi_x2 = x11, x21
    roi_y1, roi_y2 = cme, bottom_y
    cv2.rectangle(blend, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
    cv2.imshow("Blended",blend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # Display the results
    # cv2.imshow("New Flood Mask", new_flood_mask)
    # cv2.imshow("New Taxi Region", new_taxi_region)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return new_flood_mask, new_taxi_region
