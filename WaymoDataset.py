import os
import torch
from PIL import Image
import pickle
import numpy as np

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
class Camera(torch.utils.data.Dataset): #카메라하나당 하나씩
    def __init__(self,root,imgs,anno,extrinsic,intrinsic):
        self.imgs=imgs
        self.anno=anno
        self.extrinsic=extrinsic
        self.intrinsic=intrinsic
        self.root=root
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.root, self.imgs[idx])
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
        # image_id = torch.tensor(self.imgs[idx])
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(len(obj_ids,), dtype=torch.int64)
        target={}
        target["boxes"] = boxes
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = self.imgs[idx]
        target["area"] = area
        target["iscrowd"] = iscrowd
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target) 
        return img, target  #target은 GT 박스

    def __len__(self):
        return len(self.imgs)


class Waymo2DLoader(torch.utils.data.Dataset):
    def __init__(self,root,segment):
        self.root=root
        self.sequence=segment
        # self.transforms = transforms
        img_path=root+segment+"/img/"
        imgs=list(sorted(os.listdir(img_path)))[3:]
        self.ann_path=img_path+"camera_"+segment+'.pkl'
        extrinsic=np.load(img_path+"extrinsic.npy")
        intrinsic=np.load(img_path+"intrinsic.npy")
        front_anno, front_left_anno,front_right_anno,side_left_anno,side_right_anno = self.get_annotation()
        self.FRONT=Camera(img_path,imgs[:199],front_anno,extrinsic[0],intrinsic[0])
        self.FRONT_LEFT=Camera(img_path,imgs[199:398],front_left_anno,extrinsic[1],intrinsic[1])
        self.FRONT_RIGHT=Camera(img_path,imgs[398:597],front_right_anno,extrinsic[2],intrinsic[2])
        self.SIDE_LEFT=Camera(img_path,imgs[597:796],side_left_anno,extrinsic[3],intrinsic[3])
        self.SIDE_RIGHT=Camera(img_path,imgs[796:],side_right_anno,extrinsic[4],intrinsic[4])
 
    
    def get_annotation(self):
        front_anno=[]
        front_left_anno=[]
        front_right_anno=[]
        side_left_anno=[]
        side_right_anno=[]
        with open(self.ann_path, 'rb') as f:
            groundtruth=pickle.load(f)
        for i in range(len(groundtruth)):
            if i%5==0:
                front_anno.append(groundtruth[i])
            elif i%5==1:
                front_left_anno.append(groundtruth[i])
            elif i%5==2:
                front_right_anno.append(groundtruth[i])
            elif i%5==3:
                side_left_anno.append(groundtruth[i])
            elif i%5==4:
                side_right_anno.append(groundtruth[i]) 
        return  front_anno, front_left_anno,front_right_anno,side_left_anno,side_right_anno
    
    def __getitem__(self,idx): 
        imgs=[]
        targets=[]
        img,target=self.FRONT.__getitem__(idx)
        imgs.append(img)
        targets.append(target)

        img,target=self.FRONT_LEFT.__getitem__(idx)
        imgs.append(img)
        targets.append(target)

        img,target=self.FRONT_RIGHT.__getitem__(idx)
        imgs.append(img)
        targets.append(target)

        img,target=self.SIDE_LEFT.__getitem__(idx)
        imgs.append(img)
        targets.append(target)

        img,target=self.SIDE_RIGHT.__getitem__(idx)
        imgs.append(img)
        targets.append(target)
        return imgs, targets

class Waymo3DLoader(torch.utils.data.Dataset):
    def __init__(self,root,segment):
        self.root=root
        self.sequence=segment
        self.anno=self.load_anno()
        self.pointcloudlist=list(sorted(os.listdir(root+segment)))[:-2]

    def load_anno(self):
        with open(self.root+self.sequence+"/"+self.sequence+".pkl", 'rb') as f:
            groundtruth=pickle.load(f)
        return groundtruth

    def __getitem__(self,idx):
        target=self.anno[idx]
        pointcloud=np.load(self.root+self.sequence+"/"+self.pointcloudlist[idx])
        return pointcloud, target


if __name__=="__main__":
    root="./data/waymo/waymo_processed_data/"
    sequece='segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    test=Waymo2DLoader(root,sequece)
    test1=Waymo3DLoader(root,sequece)