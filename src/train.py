import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw
import random
import tensorflow.keras as keras
import imgaug.augmenters as iaa
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config

DATASET_DIR = "../prevdataset"
MODEL_DIR = "/content/drive/My Drive/mask_rcnn_logs"
IMAGES_SAVE_PATH = "visualizations"
LOG_DIR = "/content/drive/My Drive/loss/"
TRAIN_ANNOTATIONS_VISUALIZATION_SAVE_NAME = "train_annotation.jpg"
VAL_ANNOTATIONS_VISUALIZATION_SAVE_NAME = "val_annotation.jpg"

augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  
    iaa.Affine(rotate=(-20, 20)), 
    iaa.GaussianBlur(sigma=(0.0, 2.0)), 
    iaa.Multiply((0.8, 1.2)),  
    iaa.Affine(scale=(0.8, 1.2)) 
])

class LossLogger(keras.callbacks.Callback):
    def __init__(self, log_dir="logs/"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True) 
        self.log_file = os.path.join(self.log_dir, "loss_log.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Loss", "Val_Loss"])

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, loss, val_loss])
        print(f"Epoch {epoch+1}: Loss={loss}, Val_Loss={val_loss}")
    @staticmethod
    def plot_loss(log_dir):
        log_file = os.path.join(log_dir, "loss_log.csv")
        if not os.path.exists(log_file):
            print(f"Error: The file '{log_file}' does not exist.")
            return
        epochs, train_loss, val_loss = [], [], []
        with open(log_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                epochs.append(int(row[0]))
                train_loss.append(float(row[1]))
                val_loss.append(float(row[2]))
        save_path = os.path.join(log_dir, "loss.jpg")
        plt.plot(epochs, train_loss, label="Training Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

class FloodTaxiDataset(utils.Dataset):
    
    def load_dataset(self, dataset_dir, subset):
      self.add_class("flood_vehicle", 1, "flood")
      self.add_class("flood_vehicle", 2, "taxi")
      assert subset in ["train", "val","test"]
      dataset_dir = os.path.join(dataset_dir, subset)
      try:
          with open(os.path.join(dataset_dir, "via_region_data.json")) as f:
              data = json.load(f)
      except Exception as e:
          print(f"Error loading annotations: {e}")
          return
      if '_via_img_metadata' not in data:
          print("Error: Could not find '_via_img_metadata' in JSON file")
          return          
      annotations = data['_via_img_metadata']
      print(f"Found {len(annotations)} images in annotations")
      for image_key, image_data in annotations.items():
          if image_key.startswith('_via_'):
              continue             
          filename = image_data.get('filename')
          if not filename:
              continue
          regions = image_data.get('regions', [])
          if not regions:
              continue
          shapes = []
          region_attribute_name = ""
          if subset == "val":
              region_attribute_name = "FloodVehicle"
          elif subset == "test":
              region_attribute_name = "Flood_Vehicle"
          elif subset == "train":
              region_attribute_name = "Flood Vehicle"
          for region in regions:
              try:
                  shape_attributes = region.get('shape_attributes', {})
                  region_attributes = region.get('region_attributes', {})
                  category = region_attributes.get(region_attribute_name, '')
                  if not category or category.lower() not in ['flood', 'taxi']:
                      continue
                  shape_type = shape_attributes.get('name')
                  if shape_type not in ['polyline', 'polygon']:
                      continue                  
                  points_x = shape_attributes.get('all_points_x', [])
                  points_y = shape_attributes.get('all_points_y', [])                  
                  if not points_x or not points_y:
                      continue                  
                  shapes.append({
                      'points_x': points_x,
                      'points_y': points_y,
                      'type': shape_type,
                      'category': category.lower()  
                  })
              except Exception as e:
                  print(f"Error processing region in {filename}: {e}")
                  continue
          if not shapes:
              continue         
          try:
              image_path = os.path.join(dataset_dir, filename)
              image = skimage.io.imread(image_path)
              height, width = image.shape[:2]
              
              self.add_image(
                  "flood_vehicle",
                  image_id=filename,  
                  path=image_path,
                  width=width, height=height,
                  shapes=shapes  
              )
              #print(f"Successfully loaded {filename} with {len(shapes)} annotations")
          except Exception as e:
              #print(f"Error loading image {filename}: {e}")
              continue

    def load_mask(self, image_id):
      image_info = self.image_info[image_id]
      if image_info["source"] != "flood_vehicle":
          return super(self.__class__, self).load_mask(image_id)
      info = self.image_info[image_id]
      mask = np.zeros([info["height"], info["width"], len(info["shapes"])],
                      dtype=np.uint8)
      class_ids = []
      for i, shape in enumerate(info["shapes"]):
          points_y = shape['points_y']
          points_x = shape['points_x']
          if shape['type'] == 'polyline':
              points_x = points_x + [points_x[0]]
              points_y = points_y + [points_y[0]]
          rr, cc = skimage.draw.polygon(points_y, points_x)
          rr = np.clip(rr, 0, mask.shape[0] - 1)
          cc = np.clip(cc, 0, mask.shape[1] - 1)
          mask[rr, cc, i] = 1
          if shape['category'] == 'flood':
              class_ids.append(1)
          else:  
              class_ids.append(2)
      return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def visualize_random_masks(self, dataset_size,save_name):
        random_indices = random.sample(range(dataset_size), min(4, dataset_size))
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()        
        colors = ['red', 'blue']  
        
        for idx, image_id in enumerate(random_indices):
            image = skimage.io.imread(self.image_info[image_id]['path'])           
            masks, class_ids = self.load_mask(image_id)           
            axes[idx].imshow(image)            
            for mask, class_id in zip(masks.transpose(2, 0, 1), class_ids):
                contours = skimage.measure.find_contours(mask, 0.5)               
                for contour in contours:
                    axes[idx].plot(contour[:, 1], contour[:, 0], 
                                color=colors[class_id - 1], 
                                linewidth=2,
                                alpha=0.7)
            
            axes[idx].set_title(f'Image {image_id}\nRed: Flood Region, Blue: Reference Object')
            axes[idx].axis('off')
        os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
        save_path = os.path.join(IMAGES_SAVE_PATH, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  
        plt.close() 


class FloodTaxiConfig(Config):
    """
    Configuration for training on the flood vehicle dataset.
    You can tamper with other variables, except the NAME, NUM_CLASSES,
    and IMAGES_PER_GPU
    """
    NAME = "flood_vehicle"
    NUM_CLASSES = 1 + 2 
    IMAGES_PER_GPU = 2
    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    # Initially, I had set this to 100, during first round of training.
    STEPS_PER_EPOCH = 200
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    #Initially, I had set this to 10 during  first round of training.
    VALIDATION_STEPS = 60
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.9   
    # Learning rate
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    # you may not need to alter this
    LEARNING_RATE = 0.001
        
def train_model():
    config = FloodTaxiConfig()
    config.display()
    dataset_train = FloodTaxiDataset()
    dataset_train.load_dataset(DATASET_DIR, "train")
    dataset_train.prepare()
    print(f"Training images: {len(dataset_train.image_ids)}") 
    dataset_val = FloodTaxiDataset()
    dataset_val.load_dataset(DATASET_DIR, "val")
    dataset_val.prepare()
    print(f"Validation images: {len(dataset_val.image_ids)}")
    print("Visualizing training samples...")
    dataset_train.visualize_random_masks(min(4, len(dataset_train.image_ids)),
                                         TRAIN_ANNOTATIONS_VISUALIZATION_SAVE_NAME)
    print("Visualizing validation samples...")
    dataset_val.visualize_random_masks(min(4, len(dataset_val.image_ids)),
                                       VAL_ANNOTATIONS_VISUALIZATION_SAVE_NAME)
    loss_logger = LossLogger(log_dir=LOG_DIR)
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    try:    
        last_model_path = model.find_last()
        model.load_weights(last_model_path, by_name=True)
    except Exception as p:
        print("Trained heads don't exist yet, loading coco")
        model.load_weights("mrcnn_weights/mask_rcnn_coco.h5", 
                        by_name=True, 
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                              "mrcnn_bbox", "mrcnn_mask"])
            
    #Training: heads only
    print("Training network heads...")
    model.train(dataset_train, dataset_val,
                         learning_rate=config.LEARNING_RATE,
                         epochs=25,
                         layers="heads",
                         augmentation=augmentation,
                         custom_callbacks=[loss_logger]
                         ) 
   
    #Training: fine tune all layers
    print("Fine-tuning all layers...")
    model.train(dataset_train, dataset_val,
                         learning_rate=config.LEARNING_RATE/10,
                         epochs=50,
                         layers="all",
                         augmentation=augmentation,
                         custom_callbacks=[loss_logger]                        
                         )
    return model

if __name__ == "__main__":  
    model = train_model()