import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.measure import shannon_entropy

def multi_otsu_entropy_segmentation(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(image)
    num_classes = max(2, int(np.ceil(entropy)))
    thresholds = threshold_multiotsu(image, classes=num_classes)
    segmented_image = np.digitize(image, bins=thresholds)
    return (segmented_image * (255 // (num_classes - 1))).astype(np.uint8)

def draw_contour_line(binary_image, blend_image, line_color=(0, 255, 0), alpha=0.6):
    gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    h, w, _ = binary_image.shape
    first_nonzero_points = []
    for col in range(w):
        nonzero_rows = np.where(gray[:, col] > 0)[0]
        if len(nonzero_rows) > 0:
            first_nonzero_points.append((col, nonzero_rows[0]))  # (x, y)
    drawn_image = binary_image.copy()
    for i in range(len(first_nonzero_points) - 1):
        cv2.line(drawn_image, first_nonzero_points[i], first_nonzero_points[i + 1], line_color, 2)
    blended_image = cv2.addWeighted(drawn_image, alpha, blend_image, 1 - alpha, 0)
    return blended_image

def adaptive_color_threshold(flood_rgb_values):
    mean_rgb = np.mean(flood_rgb_values, axis=0)
    std_rgb = np.std(flood_rgb_values, axis=0)
    threshold_rgb = 1.5 * std_rgb
    return mean_rgb, threshold_rgb

def create_flood_mask_from_segmentation(flood_mask, segmented_image, flood_region_only):
    flood_points = np.argwhere(flood_mask == 255)
    flood_pixel_values = [segmented_image[y, x] for y, x in flood_points]
    if flood_pixel_values:
        most_frequent_intensity = np.bincount(flood_pixel_values).argmax()
        flood_mask_new = np.where(segmented_image == most_frequent_intensity, 255, 0).astype(np.uint8)
        flood_mask_new = np.maximum(flood_mask_new, flood_mask)
        flood_pixels_in_image = np.where(flood_mask_new == 255)
        for y, x in zip(flood_pixels_in_image[0], flood_pixels_in_image[1]):
            flood_region_only[y, x] = [0, 255, 0]
        return flood_mask_new, flood_region_only
    return flood_mask, flood_region_only  

def get_largest_taxi(taxis, rois, masks, class_ids):
    largest_taxi_idx = -1
    max_area = 0
    for t_idx in taxis:
        T_y1, T_x1, T_y2, T_x2 = rois[t_idx]
        area = (T_y2 - T_y1) * (T_x2 - T_x1)
        if area > max_area:
            max_area = area
            largest_taxi_idx = t_idx
    return largest_taxi_idx

def create_roi_polygon(taxi_mask, roi_mask, T_x1, T_x2, T_y2, F_y1, F_y2):
    bottom_pixels = np.argwhere(taxi_mask)
    if bottom_pixels.size == 0:
        return None   
    bottom_y_values = {}
    for y, x in bottom_pixels:
        bottom_y_values[x] = max(bottom_y_values.get(x, y), y)
    sorted_x = sorted(bottom_y_values.keys())
    bottom_points = [(x, bottom_y_values[x]) for x in sorted_x]
    extension_y = int(T_y2 + (F_y2 - F_y1) / 4)
    roi_polygon = [
        (T_x1, bottom_points[0][1]),
        *bottom_points,
        (T_x2, bottom_points[-1][1]),
        (T_x2, extension_y),
        (T_x1, extension_y)
    ]
    return np.array(roi_polygon, dtype=np.int32)

def apply_roi_smoothing(image, roi_mask, taxi_mask):
    smoothed_image = image.copy()
    for channel in range(3):
        channel_data = image[:, :, channel]
        smoothed_channel = cv2.GaussianBlur(channel_data, (7, 7), 0)
        smoothed_image[:, :, channel] = np.where(
            roi_mask == 1,
            smoothed_channel,
            channel_data
        )
    return smoothed_image

def extract_flood_points_in_roi(roi_mask, flood_mask):
    return np.argwhere((roi_mask == 1) & (flood_mask == 1))

def mark_flood_as_taxi(flood_points_in_roi, taxi_mask, new_flood_mask):
    for y, x in flood_points_in_roi:
        if taxi_mask[y, x] == 255:
            new_flood_mask[y, x] = 0 
    return new_flood_mask

def hole_fill_mask(taxi_mask):
    if taxi_mask.dtype != np.uint8:
        taxi_mask = taxi_mask.astype(np.uint8)
    taxi_mask = np.where(taxi_mask > 0, 255, 0).astype(np.uint8) 
    kernel = np.ones((11, 11), np.uint8)    
    filled_mask = cv2.morphologyEx(taxi_mask, cv2.MORPH_CLOSE, kernel)   
    return filled_mask

def visualize_refinement(image,new_flood_mask,taxi_bbox, flood_bbox, roi_polygon):
    full_channel_mask = cv2.merge([new_flood_mask] * 3)
    blend = draw_contour_line(full_channel_mask,image)
    y1, x1, y2, x2 = taxi_bbox
    cv2.rectangle(blend, (x1, y1), (x2, y2), (0, 0, 255), 2)
    y1, x1, y2, x2 = flood_bbox
    cv2.rectangle(blend, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.polylines(blend, roi_polygon, isClosed=True, color=(255, 255, 255), thickness=2)
    return blend

def process_detection_results(model_results, image):
    rois = model_results[0]['rois']
    masks = model_results[0]['masks']
    class_ids = model_results[0]['class_ids']   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    all_polygons = [] 
    if len(rois) == 0:
        return None, None, None
    taxis = [i for i, cls in enumerate(class_ids) if cls == 2] 
    floods = [i for i, cls in enumerate(class_ids) if cls == 1]
    largest_taxi_idx = get_largest_taxi(taxis, rois, masks, class_ids)
    if largest_taxi_idx == -1:
        return None, None, None
    T_y1, T_x1, T_y2, T_x2 = rois[largest_taxi_idx]
    taxi_mask = masks[:, :, largest_taxi_idx]
    new_flood_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    new_taxi_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    filled_taxi_mask = hole_fill_mask(taxi_mask)
    final_y1, final_x1, final_y2, final_x2 = None, None, None, None
    for f_idx in floods:
        F_y1, F_x1, F_y2, F_x2 = rois[f_idx]
        flood_mask = masks[:, :, f_idx]
        if (
            (T_y2 <= F_y2 and T_x1 >= F_x1 and T_x1 <= F_x2) or
            (T_y2 <= F_y2 and T_x2 >= F_x1 and T_x2 <= F_x2)
        ):
            if final_y1 is None:
              final_y1, final_x1, final_y2, final_x2 = F_y1, F_x1, F_y2, F_x2
            else:
                final_y1 = min(final_y1, F_y1)
                final_x1 = min(final_x1, F_x1)
                final_y2 = max(final_y2, F_y2)
                final_x2 = max(final_x2, F_x2)
            roi_polygon = create_roi_polygon(taxi_mask, filled_taxi_mask,
                                              T_x1, T_x2, T_y2, F_y1, F_y2)
            if roi_polygon is None:
                continue
            all_polygons.append(roi_polygon)
            roi_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_polygon], 1)
            roi_mask[filled_taxi_mask == 255] = 0
            smoothed_image = apply_roi_smoothing(image, roi_mask, filled_taxi_mask)
            flood_points_in_roi = extract_flood_points_in_roi(roi_mask, flood_mask)
            if flood_points_in_roi.size > 0:
                flood_rgb_values = np.array([
                    smoothed_image[y, x] for y, x in flood_points_in_roi
                ])
                mean_flood_rgb, threshold_rgb = adaptive_color_threshold(flood_rgb_values)
                flood_region_only = np.zeros(smoothed_image.shape, dtype=np.uint8)                
                for y in range(image.shape[0]):
                    for x in range(image.shape[1]):
                        if roi_mask[y, x] == 1 and taxi_mask[y, x] == 0:
                            pixel_rgb = smoothed_image[y, x]
                            flood_region_only[y, x] = smoothed_image[y, x]
                            color_distance = np.linalg.norm(pixel_rgb - mean_flood_rgb)
                            if (
                                abs(pixel_rgb[0] - mean_flood_rgb[0]) <= threshold_rgb[0] and
                                abs(pixel_rgb[1] - mean_flood_rgb[1]) <= threshold_rgb[1] and
                                abs(pixel_rgb[2] - mean_flood_rgb[2]) <= threshold_rgb[2]
                            ):
                                new_flood_mask[y, x] = 255                             
                segmented_image = multi_otsu_entropy_segmentation(flood_region_only)
                new_flood_mask, highlighted_image = \
                  create_flood_mask_from_segmentation(new_flood_mask,
                                                       segmented_image , 
                                                       flood_region_only)
    new_taxi_mask[T_y1:T_y2, T_x1:T_x2] = filled_taxi_mask[T_y1:T_y2, T_x1:T_x2]
    combined_polygon = all_polygons[0]
    for poly in all_polygons[1:]:
        combined_polygon = combined_polygon.union(poly)
    visual = visualize_refinement(image,new_flood_mask,rois[largest_taxi_idx],
                         (final_y1, final_x1, final_y2, final_x2),[combined_polygon])
    return new_taxi_mask, (T_y1, T_x1, T_y2, T_x2), new_flood_mask, visual


def extract_subimage_from_bbox(image, bbox):
    y1, x1, y2, x2 = bbox
    subimage = image[y1:y2, x1:x2]    
    print(f"Extracted Subimage Shape: {subimage.shape}")
    return subimage