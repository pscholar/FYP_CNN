import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import matplotlib.pyplot as plt
from train import FloodTaxiConfig
import post_process as pp
import depth_extractor as dp

CLASS_NAMES = ['BG', 'flood', 'taxi']
REFERENCE_IMAGE_ONE = "resources/Reference_Taxi_Body_Outline.jpg"
REFERENCE_IMAGE_TWO = "resources/Reference_Taxi_Body_Outline_Flipped.jpg"
REFERENCE_JSON_ONE = "resources/Taxi_Reference_Masks.json"
REFERENCE_JSON_TWO =  "resources/Taxi_Reference_Masks_Flipped.json"
OUT_DIR = "results"
MASK_RCNN_RESULTS = "detections.jpg"
REFINEMENT = "refinement.jpg"
SUBIMAGE = "subimage.jpg"
MATCHES = "matches.jpg"
BLENDED_IMAGE = "registered.jpg"
TAXI_BODY = "taximask.jpg"
WARPED_FLOOD = "floodmask.jpg"
DEPTH_RESULTS = "depth.jpg"
PIXEL_HEIGHT = 2.65
BASELINE =  [(0,767),(1831,767)]
BLUE_SEA_COLOR = (250, 180, 80)
FILE = "tests/631.jpg"

def get_ax(rows=1, cols=1, size=8):
    fig = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig

def get_output_path(file_name):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    output_path = os.path.join(OUT_DIR, file_name)
    return output_path

def save_plot_image(file_name,image,title,show = False):
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

class InferenceConfig(FloodTaxiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

model = mrcnn.model.MaskRCNN(mode="inference",
                             config=InferenceConfig(),
                             model_dir=os.getcwd())

model.load_weights(filepath="mrcnn_weights\mask_rcnn_flood_vehicle_0030.h5",
                   by_name=True)

image = cv2.imread(FILE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Detecting Flood and Taxi Regions............")
r = model.detect([image], verbose=0)
rt = r[0]
fig = get_ax(rows=1, cols=1, size=8)
mrcnn.visualize.display_instances(image=image,
                                  boxes=rt['rois'],
                                  masks=rt['masks'],
                                  class_ids=rt['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=rt['scores'],
                                  title="Mask RCNN Detection Results",
                                  figAx= fig)
save_file_path = get_output_path(MASK_RCNN_RESULTS)
plt.tight_layout()
plt.savefig(save_file_path,dpi=300)
plt.close()
print(f"Saved Detections to: {save_file_path}")

new_taxi_mask, bbox, flood_mask,refinement = pp.process_detection_results(r,image)
save_plot_image(REFINEMENT,refinement,"Selected Taxi and Flood Pixels",True)
subimage = pp.extract_subimage_from_bbox(image,bbox)
subimage =  cv2.cvtColor(subimage, cv2.COLOR_RGB2BGR) 
save_plot_image(SUBIMAGE,subimage, "Extracted Subimage using Taxi Bounding Box")

ref1 = cv2.imread(REFERENCE_IMAGE_ONE)
ref2 = cv2.imread(REFERENCE_IMAGE_TWO)
best_ref, homography_matrix, flag,matched_img = dp.get_homography_matrix(ref1,ref2,subimage)
save_plot_image(MATCHES,matched_img,"Correspondences between Reference Image and Scene Image")
print(f"Homography Matrix: {homography_matrix}")

flood_mask = dp.convert_flood_mask_to_color(flood_mask,BLUE_SEA_COLOR)
blended_image, warped_flood_mask =  \
      dp.warp_flood_over_reference_image(best_ref,flood_mask,
                                                      homography_matrix, BLUE_SEA_COLOR , 
                                                      alpha1=.5,alpha2=1.0)
save_plot_image(BLENDED_IMAGE ,blended_image,"Flood Region Registered to the Reference Image")

if flag < 0:
    taxi_mask_json_path = REFERENCE_JSON_TWO 
    
else:
    taxi_mask_json_path = REFERENCE_JSON_ONE

ref_taxi_body_mask, _ = dp.parse_json_to_masks(taxi_mask_json_path, best_ref.shape)   
ref_taxi_body_mask = cv2.cvtColor(ref_taxi_body_mask, cv2.COLOR_BGR2GRAY)
_, ref_taxi_body_mask = cv2.threshold(ref_taxi_body_mask, 10, 255, cv2.THRESH_BINARY)
save_plot_image(TAXI_BODY,ref_taxi_body_mask,"Mask of Taxi Body from Best Matching Reference Image")

warped_flood_mask = cv2.cvtColor(warped_flood_mask, cv2.COLOR_BGR2GRAY)
_, warped_flood_mask =  cv2.threshold(warped_flood_mask, 10, 255, cv2.THRESH_BINARY)
save_plot_image(WARPED_FLOOD,warped_flood_mask,"Mask of Flood Region in Registered Image")

average_depth, depths, blended_image = dp.get_flood_depth(blended_image,
                                                                       ref_taxi_body_mask,
                                                                       warped_flood_mask,
                                                                      BASELINE,PIXEL_HEIGHT)
save_plot_image(DEPTH_RESULTS,blended_image,"Estimated Depth",True)

print(f"Estimated Average Depth: {average_depth} mm")



