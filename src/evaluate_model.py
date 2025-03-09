import numpy as np
import tensorflow as tf
from mrcnn import utils
import mrcnn.model as modellib
from train import FloodTaxiConfig, FloodTaxiDataset

DATASET_DIR = "../dataset"
MODEL_DIR = "/content/drive/My Drive/mask_rcnn_logs"

class InferenceConfig(FloodTaxiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

def compute_batch_ap(image_ids):
    APs = []
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id)
        results = model.detect([image], verbose=0)
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    return APs

config = InferenceConfig()
config.display()
DEVICE = "/cpu:0"
dataset_train = FloodTaxiDataset()
dataset_train.load_dataset(DATASET_DIR, "train")
dataset_train.prepare()
print(f"Training images: {len(dataset_train.image_ids)}  Classes: {dataset_train.class_names}") 
dataset_val = FloodTaxiDataset()
dataset_val.load_dataset(DATASET_DIR, "val")
dataset_val.prepare()
print(f"Validation images: {len(dataset_val.image_ids)} Classes: {dataset_val.class_names}")
dataset_test = FloodTaxiDataset()
dataset_test.load_dataset(DATASET_DIR, "test")
dataset_test.prepare()
print(f"Test images: {len(dataset_test.image_ids)} Classes: {dataset_test.class_names}")
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    #change the weight file name appropriately
    model.load_weights("mask_rcnn_flood_vehicle_0033.h5", by_name=True)
    dset =  [dataset_val, dataset_test, dataset_train]
    dset_names = ["Validation Set","Test Set","Train Set"]
    i = 0
    for dataset in dset:
        APs = compute_batch_ap(dataset.image_ids)
        mean_AP = np.mean(APs)
        print(f"{dset_names[i]}: mAP @ IoU=50:{mean_AP}")
        i += 1


