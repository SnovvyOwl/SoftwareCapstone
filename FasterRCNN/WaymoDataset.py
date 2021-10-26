import os
import numpy as np
import torch
from PIL import Image
import pickle
from multiprocessing import Queue, Process, Pool

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']

class WaymoDataset(torch.utils.data.Dataset):
    def __init__(self,root,segment):
        self.root=root
        self.segment=segment
        self.img_path=self.root+segment+"/img/"
        self.imgs=list(sorted(os.listdir(self.img_path)))[3:]
        self.ann_path=self.img_path+"camera_"+segment+'.pkl'
        self.front_imgs=self.imgs[:199]
        self.front_left_imgs=self.imgs[199:398]
        self.front_right_imgs=self.imgs[398:597]
        self.side_left_imgs=self.imgs[597:796]
        self.side_right_imgs=self.imgs[796:]
        self.front_anno=[]
        self.front_left_anno=[]
        self.front_right_anno=[]
        self.side_left_anno=[]
        self.side_right_anno=[]
        self.anno=self.get_annotation()
    
    def get_annotation(self):
        anno=[]
        with open(self.ann_path, 'rb') as f:
            groundtruth=pickle.load(f)
        for i in range(len(groundtruth)):
            if i%5==0:
                self.front_anno.append(groundtruth[i])
            elif i%5==1:
                self.front_left_anno.append(groundtruth[i])
            elif i%5==2:
                self.front_right_anno.append(groundtruth[i])
            elif i%5==3:
                self.side_left_anno.append(groundtruth[i])
            elif i%5==4:
                self.side_right_anno.append(groundtruth[i])   
        
        return self.front_anno+self.front_left_anno+self.front_right_anno+self.side_left_anno+self.side_right_anno
    
    def __getitem__(self,idx): 
        img_path = os.path.join(self.img_path, self.imgs[idx])
        obj_ids=[]
        img=Image.open(img_path).convert("RGB")
        boxes = torch.as_tensor(self.anno[idx]["ann"]["bboxes"], dtype=torch.float32)
        for label in self.anno[idx]["ann"]["labels"]:
            if label=='unknown':
                obj_ids.append(0)
            elif label=='Vehicle':
                obj_ids.append(1)
            elif label=='Pedestrian':
                obj_ids.append(2)
            elif label=='Sign':
                obj_ids.append(3)
            elif label=='Cyclist':
                obj_ids.append(4)
        num_objs = len(obj_ids)
        image_id = torch.tensor([idx])
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(len(obj_ids,), dtype=torch.int64)
        target={}
        target["boxes"] = boxes
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target
        
if __name__=="__main__":
    root="./PV_R-CNN/data/waymo/waymo_processed_data/"
    sequece='segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    test=WaymoDataset(root,sequece)
    test.__getitem__(800)
