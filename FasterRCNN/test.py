from re import S
from mmcv import image
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import numpy as np
import cv2
from mmdet.apis.inference import show_result_pyplot
from PIL import Image
import matplotlib.pyplot as plt
from numpy.core.records import array

labels_to_names_seq = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                       11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                       21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                       31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                       41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                       51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'sofa', 58: 'pottedplant', 59: 'bed', 60: 'diningtable',
                       61: 'toilet', 62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
                       71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class FastRCNN_eval(object):
    def __init__(self,config,checkpoint,rootdir):
        self.cfg=config
        self.sequence_list = os.listdir(rootdir)
        self.model=init_detector(config, checkpoint, device='cuda:0')
        
    def save_result(self,sequence):
        image_list = os.listdir("./data/waymo/waymo_processed_data/"+sequence+"/img/")
        for file in image_list:
            if file[-1] == "g":
                img_name = "./data/waymo/waymo_processed_data/"+sequence+"/img/"+file
                result=inference_detector(self.model, img_name)
                self.model.show_result(img_name, result, score_thr=0.3,show=False,wait_time=0,win_name=file,bbox_color=(72, 101, 241),text_color=(72, 101, 241),out_file="./FasterRCNN/output/"+file[0:-4]+".jpg")
                np.save("./FasterRCNN/output/"+file[0:-4]+".npy",result)

if __name__ == "__main__":
    # Choose to use a config and initialize the detector
    config = './FasterRCNN/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = './FasterRCNN/checkpoints/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')
    rootdir = "./data/waymo/waymo_processed_data/"
    sequence_list = os.listdir(rootdir)
    for sequence in sequence_list:
        image_list = os.listdir(
            "./data/waymo/waymo_processed_data/"+sequence+"/img/")
        for file in image_list:
            if file[-1] == "g":
                img_name = "./data/waymo/waymo_processed_data/"+sequence+"/img/"+file
                result=inference_detector(model, img_name)
                model.show_result(img_name, result, score_thr=0.3,show=False,wait_time=0,win_name=file,bbox_color=(72, 101, 241),text_color=(72, 101, 241),out_file="./FasterRCNN/output/"+file[0:-4]+".jpg")
                np.save("./FasterRCNN/output/"+file[0:-4]+".npy",result)

