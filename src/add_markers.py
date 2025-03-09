
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

MAX_DISPLAY_SIZE = 1080
points1, points2 = [], []


def resize_image(image):
    height, width = image.shape[:2]
    max_dim = max(height, width)
    if max_dim > MAX_DISPLAY_SIZE:
        scaling_factor = MAX_DISPLAY_SIZE / float(max_dim)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image_resized, scaling_factor
    return image, 1.0

def select_points(event, x, y, flags, param):
    global points1, points2, selecting_first_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if selecting_first_image:
            points1.append((x, y))
            cv2.circle(image1_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select points in first image", image1_display)
        else:
            points2.append((x, y))
            cv2.circle(image2_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select points in second image", image2_display)

def get_homography(img1, img2):
    global image1, image2, image1_display, image2_display, selecting_first_image
    image1, image2 = img1.copy(), img2.copy()
    image1_display, scale1 = resize_image(image1)
    image2_display, scale2 = resize_image(image2)
    selecting_first_image = True
    cv2.imshow("Select points in first image", image1_display)
    cv2.setMouseCallback("Select points in first image", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    selecting_first_image = False
    cv2.imshow("Select points in second image", image2_display)
    cv2.setMouseCallback("Select points in second image", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(points1) < 4 or len(points2) < 4:
        print("Error: Select at least 4 points in both images!")
        return None, None
    pts1 = np.array(points1, dtype=np.float32) * (1 / scale1)
    pts2 = np.array(points2, dtype=np.float32) * (1 / scale2)
    homography_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC,2.0)
    print("Homography Matrix:\n", homography_matrix)
    matched_img = draw_matches(image1, pts1, image2, pts2)
    _, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(matched_img,cv2.COLOR_BGR2RGB))
    plt.title("Matched Points")
    plt.show()
    height, width, _ = img2.shape
    warped_img = cv2.warpPerspective(img1, homography_matrix, (width, height))
    overlay = cv2.addWeighted(img2, 0.5, warped_img, 0.5, 0)
    _, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
    plt.title("Warped Image Overlay")
    plt.show()
    return homography_matrix, matched_img

def draw_matches(img1, pts1, img2, pts2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = img1
    result[:h2, w1:w1 + w2] = img2
    for p1, p2 in zip(pts1, pts2):
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]) + w1, int(p2[1])) 
        cv2.line(result, pt1, pt2, (0, 255, 0), 2)  
        cv2.circle(result, pt1, 5, (0, 0, 255), -1)  
        cv2.circle(result, pt2, 5, (0, 0, 255), -1)      
    return result

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

def apply_mask(image, mask):
    mask = mask.astype(np.uint8)
    if len(mask.shape) == 2:  
        mask = cv2.merge([mask, mask, mask])
    if image.dtype != mask.dtype:
        mask = mask.astype(image.dtype)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def warp_and_blend(ref_image, input_image, mask, homography, flag = 1, feather_size=3):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    if flag < 0:
        homography = np.linalg.inv(homography)
    height, width = input_image.shape[:2]
    warped_ref = cv2.warpPerspective(ref_image, homography, (width, height))
    warped_mask = cv2.warpPerspective(binary_mask, homography, (width, height))
    soft_mask = cv2.GaussianBlur(warped_mask, (feather_size, feather_size), 0)
    soft_mask = soft_mask.astype(np.float32) / 255.0
    soft_mask_3ch = cv2.merge([soft_mask] * 3)
    blended_image = (soft_mask_3ch * warped_ref) + ((1 - soft_mask_3ch) * input_image)
    blended_image = blended_image.astype(np.uint8)
    return blended_image

if __name__ == "__main__":
  ref_img_path = "resources/Reference_Taxi_Body_Outline.jpg"  
  reference_image = cv2.imread(ref_img_path)
  input_img_path = "resources/to12.jpg"  
  input_image = cv2.imread(input_img_path)
  H, matched_img = get_homography(reference_image, input_image)
  taxi_mask, marker_mask = parse_json_to_masks("resources/Taxi_Reference_Masks.json", reference_image.shape)
  blended_image = warp_and_blend(reference_image,input_image,marker_mask,H,1,3)
  output_path = "resources/to12_Marked.jpg"
  cv2.imwrite(output_path,blended_image,[cv2.IMWRITE_JPEG_QUALITY, 100])
  fig, ax = plt.subplots()
  ax.imshow(cv2.cvtColor(blended_image,cv2.COLOR_BGR2RGB))
  plt.title("Taxi Warp")
  plt.tight_layout()
  plt.show()