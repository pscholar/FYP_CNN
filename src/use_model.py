import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import matplotlib.pyplot as plt
from train import FloodTaxiConfig

CLASS_NAMES = ['BG', 'flood', 'taxi']
IMAGES_SAVE_PATH = "visualizations"

def get_ax(rows=1, cols=1, size=8):
    fig = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig

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
# change the image path accordinly
image = cv2.imread("545.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r = model.detect([image], verbose=0)
r = r[0]
fig = get_ax(rows=1, cols=1, size=8)
mrcnn.visualize.display_instances(image=image,
                                  boxes=r['rois'],
                                  masks=r['masks'],
                                  class_ids=r['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=r['scores'],
                                  figAx= fig)
os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
save_path = os.path.join(IMAGES_SAVE_PATH, "predicted.jpg")
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches="tight") 
plt.show() 
plt.close()


