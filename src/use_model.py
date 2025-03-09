import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import matplotlib.pyplot as plt
from train import FloodTaxiConfig
import post_process

CLASS_NAMES = ['BG', 'flood', 'taxi']
IMAGES_SAVE_PATH = "visualizations"

def get_ax(rows=1, cols=1, size=8):
    fig = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig

def get_files_in_directory(directory_path):
    if not os.path.isdir(directory_path):
        raise ValueError(f"Invalid directory: {directory_path}")
    
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    return files, len(files)

class InferenceConfig(FloodTaxiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

model = mrcnn.model.MaskRCNN(mode="inference",
                             config=InferenceConfig(),
                             model_dir=os.getcwd())
#change the name of the weights accordingly
model.load_weights(filepath="mask_rcnn_flood_vehicle_0030.h5",
                   by_name=True)
file = "631.jpg"
image = cv2.imread(file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r = model.detect([image], verbose=0)
rt = r[0]
fig = get_ax(rows=1, cols=1, size=8)
mrcnn.visualize.display_instances(image=image,
                                  boxes=rt['rois'],
                                  masks=rt['masks'],
                                  class_ids=rt['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=rt['scores'],
                                  title=file,
                                  figAx= fig)
plt.tight_layout()
plt.show()
new_taxi_mask, bbox, new_flood_mask = post_process.process_detection_results(r,image)
subimage = post_process.extract_subimage_from_bbox(image,bbox)
subimage =  cv2.cvtColor(subimage, cv2.COLOR_RGB2BGR) 
cv2.namedWindow('Extracted Taxi', cv2.WINDOW_NORMAL)
cv2.imshow("Extracted Taxi",subimage) 
cv2.waitKey(0)
cv2.destroyAllWindows()
# change the image path accordinly
# DIR = "test"
# test_files, _ = get_files_in_directory(DIR)
# for file in test_files:   
#   image = cv2.imread(file)
#   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#   r = model.detect([image], verbose=0)
#   r = r[0]
#   fig = get_ax(rows=1, cols=1, size=8)
#   mrcnn.visualize.display_instances(image=image,
#                                     boxes=r['rois'],
#                                     masks=r['masks'],
#                                     class_ids=r['class_ids'],
#                                     class_names=CLASS_NAMES,
#                                     scores=r['scores'],
#                                     title=file,
#                                     figAx= fig)
  #os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
  #save_path = os.path.join(IMAGES_SAVE_PATH, "predicted.jpg")
  #plt.tight_layout()
  #plt.savefig(save_path, dpi=300, bbox_inches="tight") 
  #plt.show() 
  #plt.close()


