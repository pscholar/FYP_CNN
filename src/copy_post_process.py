import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.measure import shannon_entropy

def multi_otsu_segmentation(image, num_classes=3):
    """Segment image using Multi-Otsu thresholding."""
    if len(image.shape) == 3:  # Convert to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresholds = threshold_multiotsu(image, classes=num_classes)
    segmented_image = np.digitize(image, bins=thresholds)

    return (segmented_image * (255 // (num_classes - 1))).astype(np.uint8)

def multi_otsu_entropy_segmentation(image):
    """Segment image using Multi-Otsu thresholding with entropy-based class selection."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    entropy = shannon_entropy(image)
    num_classes = max(2, int(np.ceil(entropy)))  # Ensure at least 2 classes

    thresholds = threshold_multiotsu(image, classes=num_classes)
    segmented_image = np.digitize(image, bins=thresholds)

    return (segmented_image * (255 // (num_classes - 1))).astype(np.uint8)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_silhouette_segmentation(image, min_k=2, max_k=6):
    """Segment image using K-Means clustering with automatic K selection."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pixels = image.reshape(-1, 1).astype(np.float32)

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixels)
        score = silhouette_score(pixels, kmeans.labels_)

        if score > best_score:
            best_score = score
            best_k = k

    # Final K-Means clustering with best K
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(pixels)
    segmented_image = kmeans.labels_.reshape(image.shape)

    return (segmented_image * (255 // (best_k - 1))).astype(np.uint8)

def slic_superpixel_segmentation(image, num_segments=200, compactness=10):
    """
    Segment image using SLIC superpixels.
    
    Parameters:
    - image: cv2 image (BGR format).
    - num_segments: Approximate number of superpixels.
    - compactness: Balance between color similarity and spatial proximity.
    
    Returns:
    - mask: Binary mask of superpixels.
    - segmented_image: Image with superpixels outlined.
    """
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=int(np.sqrt(image.shape[0] * image.shape[1] / num_segments)), ruler=compactness)
    slic.iterate(10)  # Run 10 iterations

    mask = slic.getLabelContourMask()  # Get superpixel boundaries
    labels = slic.getLabels()  # Get the labels for each superpixel

    # Overlay mask onto the image
    segmented_image = image.copy()
    segmented_image[mask == 255] = [0, 0, 255]  # Mark boundaries in red

    return labels, segmented_image

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


def adaptive_color_threshold(flood_rgb_values):
    """
    Computes an adaptive color threshold based on mean RGB values and standard deviation.
    
    Parameters:
        flood_rgb_values (numpy.ndarray): Array of flood region RGB values.

    Returns:
        mean_rgb (numpy.ndarray): Mean RGB color vector.
        threshold_rgb (numpy.ndarray): RGB threshold values for segmentation.
    """
    # Compute mean RGB vector
    mean_rgb = np.mean(flood_rgb_values, axis=0)

    # Compute standard deviation in each channel
    std_rgb = np.std(flood_rgb_values, axis=0)

    # Set adaptive threshold using 1.25 * standard deviation
    threshold_rgb = 1.25 * std_rgb

    print("Mean RGB:", mean_rgb)
    print("Threshold RGB:", threshold_rgb)

    return mean_rgb, threshold_rgb


def process_detection_results(model_results, image):
    """
    Processes Mask R-CNN detection results to analyze taxi-flood overlap,
    define a polygonal region of interest (ROI), filter flood pixels using adaptive RGB color similarity,
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
    i = 0
    # r,g,b = cv2.split(image)
    # r_eq = cv2.equalizeHist(r)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Ensure OpenCV format (BGR)
    # hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    # eq_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow("original",image)
    #cv2.imshow("Histograme Equalized",eq_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #image = eq_image
    save = False;
    rep = None
    saba = False;
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
                    (T_x1, bottom_points[0][1]),  # Leftmost bottom pixel
                    *bottom_points,  # All bottom pixels
                    (T_x2, bottom_points[-1][1]),  # Rightmost bottom pixel
                    (T_x2, extension_y),  # Extended bottom right
                    (T_x1, extension_y)  # Extended bottom left
                ]

                roi_polygon = np.array(roi_polygon, dtype=np.int32)
                if not save:
                    rep = roi_polygon
                    save = True
                # Create ROI mask (polygonal region)
                roi_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
                cv2.fillPoly(roi_mask, [roi_polygon], 1)  # Fill polygonal region

                # Exclude taxi mask from ROI
                roi_mask[taxi_mask == 1] = 0

                # Apply Gaussian smoothing only in ROI
                smoothed_image = image.copy()
                for channel in range(3):  # Process each color channel
                    channel_data = image[:, :, channel]
                    smoothed_channel = cv2.GaussianBlur(channel_data, (7, 7), 0)
                    smoothed_image[:, :, channel] = np.where(
                        roi_mask == 1,  # Apply smoothing only in ROI
                        smoothed_channel,
                        channel_data
                    )

                # Get flood pixels in ROI
                flood_points_in_roi = np.argwhere((roi_mask == 1) & (flood_mask == 1))

                if flood_points_in_roi.size > 0:
                    # Extract RGB values of flood pixels in the ROI (from smoothed image)
                    flood_rgb_values = np.array([
                        smoothed_image[y, x] for y, x in flood_points_in_roi
                    ])

                    # Compute mean RGB and threshold
                    mean_flood_rgb, threshold_rgb = adaptive_color_threshold(flood_rgb_values)
                    image_only = np.zeros(smoothed_image.shape, dtype=np.uint8)
                    # Iterate over all pixels in ROI
                    for y in range(image.shape[0]):
                        for x in range(image.shape[1]):
                            if roi_mask[y, x] == 1 and taxi_mask[y, x] == 0:
                                pixel_rgb = smoothed_image[y, x]
                                image_only[y,x] = smoothed_image[y,x]
                                # Compute Euclidean distance from the mean color
                                color_distance = np.linalg.norm(pixel_rgb - mean_flood_rgb)
                                # Check if the pixel is within the RGB threshold box
                                if (
                                    abs(pixel_rgb[0] - mean_flood_rgb[0]) <= threshold_rgb[0] and
                                    abs(pixel_rgb[1] - mean_flood_rgb[1]) <= threshold_rgb[1] and
                                    abs(pixel_rgb[2] - mean_flood_rgb[2]) <= threshold_rgb[2]
                                ):
                                    if not saba and i < 20:
                                      print(f"points:{(y,x)} value: {pixel_rgb} ")
                                      i += 1
                                    new_flood_mask[y, x] = 255  # Mark as flood
                    cv2.imshow("Image Only",image_only)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    # Convert to grayscale properly
                    # seg1 = multi_otsu_segmentation(image_only, num_classes=3)
                    # cv2.imshow("Multi-Otsu (3 classes)", seg1)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    seg2 = multi_otsu_entropy_segmentation(image_only)
                    cv2.imshow("Multi-Otsu + Entropy", seg2)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    # seg3 = kmeans_silhouette_segmentation(image_only)
                    # # Display the results
                    # cv2.imshow("K-Means + Silhouette", seg3)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    ##from post process:
                #     segmented_image = multi_otsu_entropy_segmentation(flood_region_only)
                # # Optional visualization
                # cv2.imshow("Image Only", flood_region_only)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # segmented_image = multi_otsu_entropy_segmentation(flood_region_only)
                # cv2.imshow("Multi-Otsu + Entropy", segmented_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # new_flood_mask, highlighted_image = create_flood_mask_from_segmentation(new_flood_mask,segmented_image , flood_region_only)
                # cv2.imshow("New Flood Mask", new_flood_mask)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imshow("Highlighted Image", highlighted_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                saba = True

                # Extract taxi region pixels
                new_taxi_region[T_y1:T_y2, T_x1:T_x2] = image[T_y1:T_y2, T_x1:T_x2] * taxi_mask[T_y1:T_y2, T_x1:T_x2, None]


    # Display results
    #cv2.imshow("Original New Flood Mask", new_flood_mask)
    #cv2.imshow("New Taxi Region", new_taxi_region)
    # Apply blending
    mask = cv2.merge([new_flood_mask] * 3)
    blend = draw_contour_line(mask,image)
    y1, x1, y2, x2 = rois[taxis[0]]
    cv2.rectangle(blend, (x1, y1), (x2, y2), (0, 0, 255), 2)
    y1, x1, y2, x2 = rois[floods[0]]
    cv2.rectangle(blend, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.polylines(blend, [rep], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.imshow("Blended",blend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return new_flood_mask, new_taxi_region
