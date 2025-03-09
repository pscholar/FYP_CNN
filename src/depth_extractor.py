import json
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import post_process
import sift3

blue_sea_color = (180, 100, 130)

def extract_bounding_boxes(json_path, image_size):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    image_data = list(data["_via_img_metadata"].values())[0]
    regions = image_data["regions"]
    
    taxi_bbox = None
    flood_mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)  # Create a 3-channel blank image   
    for region in regions:
        attributes = region["region_attributes"]
        shape = region["shape_attributes"]
        
        if attributes.get("Test_Sample") == "taxi":
            x1, y1 = shape["x"], shape["y"]
            x2, y2 = x1 + shape["width"], y1 + shape["height"]
            taxi_bbox = (y1, x1, y2, x2)
        elif attributes.get("Test_Sample") == "flood":
            flood_points = np.array(list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32)
            cv2.fillPoly(flood_mask, [flood_points], blue_sea_color)  # Fill flood region with blue color
    
    return taxi_bbox, flood_mask

def get_homography_matrix(ref1,ref2,target, vis = True):   
    best_ref, keypoints1, keypoints2, good_matches, flag = sift3.find_best_reference(ref1, ref2, target)
    if(vis):
      sift3.visualize_matches(best_ref, target, keypoints1, keypoints2, good_matches)
    H, mask = sift3.compute_homography(best_ref, keypoints1, keypoints2, good_matches)
    return best_ref,H,flag 

def warp_entire_image(target, input_image, homography_matrix, 
                      blue_sea_color, alpha1=0.5, alpha2=0.5):
    height, width, _ = target.shape
    warped_img = cv2.warpPerspective(input_image, homography_matrix, (width, height))

    # Define the color range for the blue sea region (pass the blue_sea_color as a parameter)
    lower_blue = np.array([blue_sea_color[0] - 10, blue_sea_color[1] - 10, blue_sea_color[2] - 10])
    upper_blue = np.array([blue_sea_color[0] + 10, blue_sea_color[1] + 10, blue_sea_color[2] + 10])
    
    # Mask out only the blue sea region in the flood mask (warped image)
    blue_mask = cv2.inRange(warped_img, lower_blue, upper_blue)

    # Convert images to float for better blending precision
    target_float = target.astype(np.float32) / 255.0
    warped_img_float = warped_img.astype(np.float32) / 255.0
    
    # Use the mask to blend only the overlapping regions
    blended_img = target_float * (1 - blue_mask[:, :, None] / 255.0) + warped_img_float * (blue_mask[:, :, None] / 255.0)
    
    # Convert back to uint8
    blended_img = (blended_img * 255).astype(np.uint8)

    # Show the result
    _, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
    plt.title("Warped Image Overlay with Blue Sea Masked Blending")
    plt.show()



# Example usage
json_path = "To Embed/test_image_via_data.json"
image_path = "To Embed/624_Marked.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape
taxi_bbox, flood_mask = extract_bounding_boxes(json_path,(height,width))
cv2.namedWindow('Flood mask', cv2.WINDOW_NORMAL)
cv2.imshow("Flood mask", flood_mask) 
cv2.waitKey(0)
cv2.destroyAllWindows()
subimage = post_process.extract_subimage_from_bbox(image,taxi_bbox)
print(f"Image Shape: {subimage.shape}")
#subimage =  cv2.cvtColor(subimage, cv2.COLOR_RGB2BGR) 
cv2.namedWindow('Extracted Taxi', cv2.WINDOW_NORMAL)
cv2.imshow("Extracted Taxi",subimage) 
cv2.waitKey(0)
cv2.destroyAllWindows()
ref1 = cv2.imread("To Embed/Reference_Taxi_Body_Outline.jpg")
ref2 = cv2.imread("To Embed/Reference_Taxi_Body_Outline_Flipped.jpg")
best_ref, homography_matrix, flag = get_homography_matrix(ref1,ref2,subimage)
print(f"Homography: {homography_matrix}",)
warp_entire_image(best_ref,flood_mask,homography_matrix, blue_sea_color 
                  ,alpha1=.5,alpha2=1.0)