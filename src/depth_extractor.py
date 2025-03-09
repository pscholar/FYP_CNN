import json
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import post_process as pp
import correspondence as corr
import os

CLASS_NAMES = ['BG', 'flood', 'taxi']
REFERENCE_IMAGE_ONE = "resources/Reference_Taxi_Body_Outline.jpg"
REFERENCE_IMAGE_TWO = "resources/Reference_Taxi_Body_Outline_Flipped.jpg"
REFERENCE_JSON_ONE = "resources/Taxi_Reference_Masks.json"
REFERENCE_JSON_TWO =  "resources/Taxi_Reference_Masks_Flipped.json"
OUT_DIR = "results"
MASK_RCNN_RESULTS = "detections.jpg"
SUBIMAGE = "subimage.jpg"
MATCHES = "matches.jpg"
BLENDED_IMAGE = "registered.jpg"
TAXI_BODY = "taximask.jpg"
WARPED_FLOOD = "floodmask.jpg"
DEPTH_RESULTS = "depth.jpg"
PIXEL_HEIGHT = 2.65
BASELINE =  [(0,767),(1831,767)]
BLUE_SEA_COLOR = (250, 180, 80)
FILE = "631.jpg"
blue_sea_color = (250, 180, 80)

def get_ax(rows=1, cols=1, size=8):
    fig = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig

def get_output_path(file_name):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    output_path = os.path.join(OUT_DIR, file_name)
    return output_path

def save_plot_image(file_name,image,title, show = False):
    save_path = get_output_path(file_name)
    fig = get_ax(rows=1, cols=1, size=8)
    _, ax = fig
    ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    ax.set_xlabel("pixels")  
    ax.set_ylabel("pixels")
    plt.tight_layout()
    plt.title(title)
    plt.savefig(save_path,dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Saved a figure to: {save_path}")

def apply_mask(image, mask):
    mask = mask.astype(np.uint8)
    if len(mask.shape) == 2:  
        mask = cv2.merge([mask, mask, mask])
    if image.dtype != mask.dtype:
        mask = mask.astype(image.dtype)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def convert_flood_mask_to_color(flood_mask,blue_sea_color=(250, 180, 80)):
    flood_mask_color = np.zeros((flood_mask.shape[0], flood_mask.shape[1], 3), dtype=np.uint8)
    flood_mask_color[flood_mask == 255] = blue_sea_color    
    return flood_mask_color

def parse_json_to_masks(json_path, img_shape):
    with open(json_path, 'r') as file:
        data = json.load(file)
    img_data = list(data["_via_img_metadata"].values())[0]   
    taxi_body_mask = np.zeros(img_shape, dtype=np.uint8)
    markers_mask = np.zeros(img_shape, dtype=np.uint8)
    for region in img_data["regions"]:
        shape = region["shape_attributes"]
        roi_type = region["region_attributes"]["SIFT_ROI"]
        if shape["name"] == "rect":
            x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
            mask = markers_mask if roi_type == "markers" else None
        elif shape["name"] == "polygon":
            points = np.array(list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32)
            mask = taxi_body_mask if roi_type == "taxi_body" else None
        else:
            continue        
        if mask is not None:
            if shape["name"] == "rect":
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)
            elif shape["name"] == "polygon":
                cv2.fillPoly(mask, [points], (255, 255, 255))
    return taxi_body_mask, markers_mask

def extract_bounding_boxes(json_path, image_size):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    image_data = list(data["_via_img_metadata"].values())[0]
    regions = image_data["regions"]
    
    taxi_bbox = None
    flood_mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)  # Create a 3-channel blank image   
    for region in regions:
        attributes = region["region_attributes"]
        shape = region["shape_attributes"]
        
        if attributes.get("Test_Sample") == "taxi":
            x1, y1 = shape["x"], shape["y"]
            x2, y2 = x1 + shape["width"], y1 + shape["height"]
            taxi_bbox = (y1, x1, y2, x2)
        elif attributes.get("Test_Sample") == "flood":
            flood_points = np.array(list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32)
            cv2.fillPoly(flood_mask, [flood_points], 255)  
    
    return taxi_bbox, flood_mask

def get_homography_matrix(ref1,ref2,target):   
    best_ref, keypoints1, keypoints2, good_matches, flag = corr.find_best_reference(ref1, ref2, target)
    matched_img = corr.visualize_matches(best_ref, target, keypoints1, keypoints2, good_matches)
    H, mask = corr.compute_homography(best_ref, keypoints1, keypoints2, good_matches)
    return best_ref,H,flag,matched_img 

def warp_flood_over_reference_image(target, input_image, homography_matrix, 
                      blue_sea_color, alpha1=0.5, alpha2=0.5):
    height, width, _ = target.shape
    warped_img = cv2.warpPerspective(input_image, homography_matrix, (width, height))
    lower_blue = np.array([blue_sea_color[0] - 10, blue_sea_color[1] - 10, blue_sea_color[2] - 10])
    upper_blue = np.array([blue_sea_color[0] + 10, blue_sea_color[1] + 10, blue_sea_color[2] + 10])
    blue_mask = cv2.inRange(warped_img, lower_blue, upper_blue)
    target_float = target.astype(np.float32) / 255.0
    warped_img_float = warped_img.astype(np.float32) / 255.0
    blended_img = target_float * (1 - blue_mask[:, :, None] / 255.0) + warped_img_float * (blue_mask[:, :, None] / 255.0)
    blended_img = (blended_img * 255).astype(np.uint8)
    return blended_img, warped_img

def get_equally_spaced_points(x_fit, y_fit):
    num_points = len(x_fit)
    if num_points < 3:
        raise ValueError("Not enough points to select three equally spaced ones.")
    idx1 = 0  
    idx2 = num_points // 2  
    idx3 = num_points - 1 
    selected_points = [
        (x_fit[idx1], y_fit[idx1]),
        (x_fit[idx2], y_fit[idx2]),
        (x_fit[idx3], y_fit[idx3])
    ]
    return selected_points

def get_flood_depth(blended_image,reference_mask,warped_flood_mask,
                    baseline,pixel_height):
    intersection = cv2.bitwise_and(reference_mask, warped_flood_mask)
    height, width = intersection.shape
    points = []
    for col in range(width):
        non_zero_pixels = np.where(intersection[:, col] > 0)[0]
        if len(non_zero_pixels) > 0:
            first_non_zero = non_zero_pixels[0]
            points.append((col, first_non_zero))
    points = np.array(points)
    if len(points) > 1:  
        x_coords = points[:, 0]  
        y_coords = points[:, 1] 
        m, c = np.polyfit(x_coords, y_coords, 1)
        x_fit = np.linspace(x_coords.min(), x_coords.max(), num=width).astype(int)
        y_fit = (m * x_fit + c).astype(int)
        x_fit = np.clip(x_fit, 0, width - 1)
        y_fit = np.clip(y_fit, 0, height - 1)
    for i in range(len(x_fit) - 1):
        cv2.line(blended_image, (x_fit[i], y_fit[i]), (x_fit[i + 1], y_fit[i + 1]), (0, 0, 255), 3)
    cv2.line(blended_image, baseline[0], baseline[1], (0, 0, 0), 3)
    selected_points = get_equally_spaced_points(x_fit, y_fit)  
    baseline_gradient = (baseline[1][1] - baseline[0][1]) / (baseline[1][0] - baseline[0][0])
    baseline_intercept = baseline[0][1] - baseline_gradient * baseline[0][0]
    height, width = blended_image.shape[:2] 
    new_coordinates = []
    for (x_fit_i, y_fit_i) in selected_points:
        y_intersection = x_fit_i * baseline_gradient  + baseline_intercept 
        y_intersection = int(round(y_intersection))
        y_intersection = max(0, min(y_intersection, height - 1))
        x_fit_i = max(0, min(x_fit_i, width - 1))
        new_coordinates.append((x_fit_i, y_intersection))
    for i in range(len(selected_points)):
        cv2.circle(blended_image, selected_points[i], 3, (255, 255, 0), -1)
    depths = []
    average_depth = 0
    for i in range(len(new_coordinates)):
        x, y = selected_points[i]  
        pixel_depth = new_coordinates[i][1] - selected_points[i][1]
        real_depth = pixel_depth * pixel_height
        depths.append(real_depth)
        average_depth += real_depth
    average_depth /=  len(new_coordinates)
    text = f"{int(average_depth)}mm"
    text_size, _ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_width = text_size[0]
    center = int(len(selected_points) / 2)
    center_x = selected_points[center][0]
    space_right = width - center_x
    space_left = center_x
    if (space_right >= space_left) or (space_right >= text_width + 2) :
        text_x = center_x + 2
    else:
        text_x  = center_x - text_width - 2 
    start_y = 100
    print((center_x , start_y))
    print(selected_points[center])
    cv2.arrowedLine(blended_image, (center_x , start_y), selected_points[center],(53, 53, 53), 2)   
    cv2.putText(blended_image, text, (text_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    for i in range(len(selected_points)):
        cv2.circle(blended_image, selected_points[i], 20, (0, 255, 255), -1)
    return average_depth,depths,blended_image

if __name__ == "__main__":
  json_path = "resources/test_image_via_data.json"
  image_path = "resources/624_Marked.jpg"
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  height, width, _ = image.shape
  taxi_bbox, flood_mask = extract_bounding_boxes(json_path,(height,width))
  subimage = pp.extract_subimage_from_bbox(image,taxi_bbox)
  subimage =  cv2.cvtColor(subimage, cv2.COLOR_RGB2BGR)
  save_plot_image(SUBIMAGE,subimage, "Extracted Subimage using Taxi Bounding Box")

  ref1 = cv2.imread(REFERENCE_IMAGE_ONE)
  ref2 = cv2.imread(REFERENCE_IMAGE_TWO)
  best_ref, homography_matrix, flag,matched_img = get_homography_matrix(ref1,ref2,subimage)
  save_plot_image(MATCHES,matched_img,"Correspondences between Reference Image and Scene Image")
  print(f"Homography Matrix: {homography_matrix}")

  flood_mask = convert_flood_mask_to_color(flood_mask,BLUE_SEA_COLOR)

  blended_image, warped_flood_mask =  \
        warp_flood_over_reference_image(best_ref,flood_mask,
                                                        homography_matrix, BLUE_SEA_COLOR , 
                                                        alpha1=.5,alpha2=1.0)
  save_plot_image(BLENDED_IMAGE ,blended_image,"Flood Region Registered to the Reference Image")

  if flag < 0:
      taxi_mask_json_path = REFERENCE_JSON_TWO 
      
  else:
      taxi_mask_json_path = REFERENCE_JSON_ONE

  ref_taxi_body_mask, _ = parse_json_to_masks(taxi_mask_json_path, best_ref.shape)   
  ref_taxi_body_mask = cv2.cvtColor(ref_taxi_body_mask, cv2.COLOR_BGR2GRAY)
  _, ref_taxi_body_mask = cv2.threshold(ref_taxi_body_mask, 10, 255, cv2.THRESH_BINARY)
  save_plot_image(TAXI_BODY,ref_taxi_body_mask,"Mask of Taxi Body from Best Matching Reference Image")

  warped_flood_mask = cv2.cvtColor(warped_flood_mask, cv2.COLOR_BGR2GRAY)
  _, warped_flood_mask =  cv2.threshold(warped_flood_mask, 10, 255, cv2.THRESH_BINARY)
  save_plot_image(WARPED_FLOOD,warped_flood_mask,"Mask of Flood Region in Registered Image")

  average_depth, depths, blended_image = get_flood_depth(blended_image,
                                                                        ref_taxi_body_mask,
                                                                        warped_flood_mask,
                                                                        BASELINE,PIXEL_HEIGHT)
  save_plot_image(DEPTH_RESULTS,blended_image,"Estimated Depth",show=True)

  print(f"Estimated Average Depth: {average_depth}mm")
 