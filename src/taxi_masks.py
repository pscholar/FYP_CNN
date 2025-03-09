# contains functions for creating a mask for markers in the
# reference image and also creating a mask for the region of the 
# reference  image that is occupied by the taxi

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

image = cv2.imread("To Embed/Reference_Taxi_Body_Outline.jpg")
taxi_mask, marker_mask = parse_json_to_masks("To Embed/Taxi_Reference_Masks.json", image.shape)
masked_image = apply_mask(image,  marker_mask)
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(masked_image,cv2.COLOR_BGR2RGB))
plt.title("Taxi Markers")
plt.show()
