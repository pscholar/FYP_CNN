import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import skimage.draw
import random
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config
#from mrcnn import visualize
DATASET_DIR = "../dataset"
MODEL_DIR = "logs"
IMAGES_SAVE_PATH = "visualizations"
LOSS_SAVE_NAME = "loss.jpg"
TRAIN_ANNOTATIONS_VISUALIZATION_SAVE_NAME = "train_annotation.jpg"
VAL_ANNOTATIONS_VISUALIZATION_SAVE_NAME = "val_annotation.jpg"
EVAL_METRICS_FIGURE_SAVE_NAME = "eval_metrics.jpg"
EVAL_METRICS_FILE_SAVE_NAME = "eval_metrics.txt"
class FloodTaxiDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
      """Load a subset of the flood vehicle dataset.
      
      Args:
          dataset_dir: Root directory of the dataset
          subset: Subset to load: train or val
      """
      # Add classes for flood vehicles
      self.add_class("flood_vehicle", 1, "flood")
      self.add_class("flood_vehicle", 2, "taxi")
      
      # Train or validation dataset?
      assert subset in ["train", "val","test"]
      dataset_dir = os.path.join(dataset_dir, subset)
      
      # Load annotations
      try:
          with open(os.path.join(dataset_dir, "via_region_data.json")) as f:
              data = json.load(f)
      except Exception as e:
          print(f"Error loading annotations: {e}")
          return
      
      # Get the actual annotations from _via_img_metadata
      if '_via_img_metadata' not in data:
          print("Error: Could not find '_via_img_metadata' in JSON file")
          return
          
      annotations = data['_via_img_metadata']
      print(f"Found {len(annotations)} images in annotations")
      
      # Process each image's annotations
      for image_key, image_data in annotations.items():
          # Skip metadata entries
          if image_key.startswith('_via_'):
              continue
              
          filename = image_data.get('filename')
          if not filename:
              continue
              
          # Get regions (annotations)
          regions = image_data.get('regions', [])
          if not regions:
              continue
              
          # Process shapes
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
                  
                  # Get category (FloodVehicle in this case)
                  category = region_attributes.get(region_attribute_name, '')
                  if not category or category.lower() not in ['flood', 'taxi']:
                      continue
                  
                  # Verify shape type and coordinates exist
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
                      'category': category.lower()  # normalize to lowercase
                  })
              except Exception as e:
                  print(f"Error processing region in {filename}: {e}")
                  continue
          
          # Skip images without valid shapes
          if not shapes:
              continue
          
          try:
              # Load image to get dimensions
              image_path = os.path.join(dataset_dir, filename)
              image = skimage.io.imread(image_path)
              height, width = image.shape[:2]
              
              self.add_image(
                  "flood_vehicle",
                  image_id=filename,  # use file name as a unique image id
                  path=image_path,
                  width=width, height=height,
                  shapes=shapes  # store both shape data and categories
              )
              print(f"Successfully loaded {filename} with {len(shapes)} annotations")
          except Exception as e:
              print(f"Error loading image {filename}: {e}")
              continue
      
      print(f"Successfully loaded {len(self.image_info)} images with annotations")

    def load_mask(self, image_id):
      """Generate instance masks for an image.
      
      Returns:
          masks: A bool array of shape [height, width, instance count] with
              one mask per instance.
          class_ids: array of class IDs of the instance masks.
      """
      # If not a flood_vehicle dataset image, delegate to parent class.
      image_info = self.image_info[image_id]
      if image_info["source"] != "flood_vehicle":
          return super(self.__class__, self).load_mask(image_id)

      # Convert shapes to a bitmap mask of shape [height, width, instance_count]
      info = self.image_info[image_id]
      mask = np.zeros([info["height"], info["width"], len(info["shapes"])],
                      dtype=np.uint8)
      class_ids = []

      for i, shape in enumerate(info["shapes"]):
          # Get points
          points_y = shape['points_y']
          points_x = shape['points_x']
          
          # For polylines, connect the first and last points to create a closed polygon
          if shape['type'] == 'polyline':
              points_x = points_x + [points_x[0]]
              points_y = points_y + [points_y[0]]
          
          # Get indexes of pixels inside the polygon and set them to 1
          rr, cc = skimage.draw.polygon(points_y, points_x)
          
          # Ensure coordinates do not exceed the mask size
          rr = np.clip(rr, 0, mask.shape[0] - 1)
          cc = np.clip(cc, 0, mask.shape[1] - 1)
          mask[rr, cc, i] = 1
          
          # Set class ID based on category
          if shape['category'] == 'flood':
              class_ids.append(1)
          else:  # taxi
              class_ids.append(2)

      return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def visualize_random_masks(self, dataset_size,save_name):
        """Visualize 4 randomly selected masks from the dataset.
        
        Args:
            dataset_size: Total number of images in the dataset
        """
        # Select 4 random image indices
        random_indices = random.sample(range(dataset_size), min(4, dataset_size))
        
        # Create a 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()
        
        colors = ['red', 'blue']  # red for flood, blue for taxi
        
        for idx, image_id in enumerate(random_indices):
            # Load image
            image = skimage.io.imread(self.image_info[image_id]['path'])           
            # Load masks and class IDs
            masks, class_ids = self.load_mask(image_id)           
            # Plot original image
            axes[idx].imshow(image)            
            # Plot all masks for this image
            for mask, class_id in zip(masks.transpose(2, 0, 1), class_ids):
                # Create mask outline
                contours = skimage.measure.find_contours(mask, 0.5)               
                # Plot each contour
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save instead of show
        plt.close()  # Close the plot to free memory


class FloodTaxiConfig(Config):
    """Configuration for training on the flood vehicle dataset."""
    NAME = "flood_vehicle"
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + flood + taxi 
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100   
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9   
    # Image size to use for training
    IMAGES_PER_GPU = 2
    VALIDATION_STEPS = 10


def train_model(dataset_dir):
    #configuring model
    config = FloodTaxiConfig()
    config.display()
    #loading training dataset
    dataset_train = FloodTaxiDataset()
    dataset_train.load_dataset("../dataset", "train")
    dataset_train.prepare()
    print(f"Training images: {len(dataset_train.image_ids)}") 
    #loading validation dataset
    dataset_val = FloodTaxiDataset()
    dataset_val.load_dataset("../dataset", "val")
    dataset_val.prepare()
    print(f"Validation images: {len(dataset_val.image_ids)}")
    #visualizing datasets
    print("Visualizing training samples...")
    dataset_train.visualize_random_masks(min(4, len(dataset_train.image_ids)),
                                         TRAIN_ANNOTATIONS_VISUALIZATION_SAVE_NAME)
    print("Visualizing validation samples...")
    dataset_val.visualize_random_masks(min(4, len(dataset_val.image_ids)),
                                       VAL_ANNOTATIONS_VISUALIZATION_SAVE_NAME)
    #create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    # Load COCO weights
    try:
        model.load_weights("mask_rcnn_coco.h5", 
                         by_name=True, 
                         exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    #Training: head only
    print("Training network heads...")
    history1 = model.train(dataset_train, dataset_val,
                         learning_rate=config.LEARNING_RATE,
                         epochs=10,
                         layers="heads")  
    #Training: fine tune all layers
    print("Fine-tuning all layers...")
    history2 = model.train(dataset_train, dataset_val,
                         learning_rate=config.LEARNING_RATE,
                         epochs=20,
                         layers="all")
    #Combine histories
    history = {}
    for k in history1.history.keys():
        history[k] = history1.history[k] + history2.history[k]  
    # Plot training metrics
    plot_training_metrics(history, LOSS_SAVE_NAME)  
    return model, history

def evaluate_model(model, dataset_dir):
    """
    Evaluate the trained model on test dataset.
    
    Args:
        model: Trained Mask R-CNN model
        dataset_dir: Root directory of the dataset
    """
    config = FloodTaxiConfig()
    # Load test dataset
    dataset_test = FloodTaxiDataset()
    dataset_test.load_dataset(dataset_dir, "test")
    dataset_test.prepare()
    print(f"Test images: {len(dataset_test.image_ids)}")
    
    # Initialize metrics
    APs, precisions, recalls, f1_scores = [], [], [], []
    y_true, y_pred = [], []
    
    # Evaluate each image
    for image_id in dataset_test.image_ids:
        # Load ground truth
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_test, config, image_id)
        
        # Run detection
        results = model.detect([image], verbose=0)
        r = results[0]
        
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                           r["rois"], r["class_ids"], r["scores"], r['masks'])
        
        # Compute precision, recall, F1
        if len(gt_class_id) > 0 and len(r['class_ids']) > 0:
            p, r, f1, _ = precision_recall_fscore_support(
                gt_class_id, r['class_ids'], average='macro')
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)           
            # Store for confusion matrix
            y_true.extend(gt_class_id)
            y_pred.extend(r['class_ids'])      
        APs.append(AP)
    
    # Print metrics
    print_evaluation_metrics(APs, precisions, recalls, f1_scores, 
                             y_true, y_pred,EVAL_METRICS_FIGURE_SAVE_NAME)

def plot_training_metrics(history,save_name):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))  
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
    save_path = os.path.join(IMAGES_SAVE_PATH, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save instead of show
    plt.close()  # Close the plot to free memory


def print_evaluation_metrics(APs, precisions, recalls, f1_scores, 
                             y_true, y_pred, save_name):
    os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
    save_path = os.path.join(IMAGES_SAVE_PATH,  EVAL_METRICS_FILE_SAVE_NAME)
    conf_matrix = confusion_matrix(y_true, y_pred);
    with open(save_path, "w") as f:
      sys.stdout = f 
      print("\nEvaluation Metrics:")
      print(f"Mean Average Precision (mAP): {np.mean(APs):.4f}")
      print(f"Mean Precision: {np.mean(precisions):.4f}")
      print(f"Mean Recall: {np.mean(recalls):.4f}")
      print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")   
      print("\nConfusion Matrix:")
      print(conf_matrix)
      sys.stdout = sys.__stdout__

    print("\nEvaluation Metrics:")
    print(f"Mean Average Precision (mAP): {np.mean(APs):.4f}")
    print(f"Mean Precision: {np.mean(precisions):.4f}")
    print(f"Mean Recall: {np.mean(recalls):.4f}")
    print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")   
    print("\nConfusion Matrix:")
    print(conf_matrix)
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Background', 'Flood', 'Taxi']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
    save_path = os.path.join(IMAGES_SAVE_PATH, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save instead of show
    plt.close()  # Close the plot to free memory


# Main execution
if __name__ == "__main__":  
    # Train model
    model, history = train_model(DATASET_DIR)  
    # Evaluate model
    evaluate_model(model, MODEL_DIR)
