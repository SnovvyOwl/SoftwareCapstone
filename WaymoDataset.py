import os
import torch
from PIL import Image
import pickle
import numpy as np
import cv2
WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
CAMMERA_NUM=5

class Camera(torch.utils.data.Dataset):  # 카메라하나당 하나씩
    def __init__(self, root, imgs, anno, camera):
        self.imgs = imgs
        self.anno = anno
        self.root = root
        self.camera = camera

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        obj_ids = []
        img = Image.open(img_path).convert("RGB")
        image = np.array(img)
        image=image[:, :, ::-1].copy() 

        boxes = torch.as_tensor(self.anno[idx]["ann"]["bboxes"], dtype=torch.float32)
        for label in self.anno[idx]["ann"]["labels"]:
            if label == 'unknown':
                obj_ids.append(0)
            elif label == 'Vehicle':
                obj_ids.append(1)
            elif label == 'Pedestrian':
                obj_ids.append(2)
            elif label == 'Sign':
                obj_ids.append(3)
            elif label == 'Cyclist':
                obj_ids.append(4)
        # image_id = torch.tensor(self.imgs[idx])
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(len(obj_ids, ), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = self.imgs[idx]
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["camera"] = self.camera

        return image, target  # target은 GT 박스

    def __len__(self):
        return len(self.imgs)


class Waymo2DLoader(torch.utils.data.Dataset):
    def __init__(self, root, segment):
        self.root = root
        self.sequence = segment
        # self.transforms = transforms
        img_path = root + segment + "/img/"
        imgs = list(sorted(os.listdir(img_path)))[3:]
        self.ann_path = img_path + "camera_" + segment + '.pkl'
        self.extrinsic = np.load(img_path + "extrinsic.npy")
        self.length=len(os.listdir(self.root+segment+'/'))-2
        self.intrinsic, self.disorted_coeff= self.make_intrinsic_mat(np.load(img_path + "intrinsic.npy"))
        front_anno, front_left_anno, front_right_anno, side_left_anno, side_right_anno = self.get_annotation()
        self.FRONT = Camera(img_path, imgs[:self.length], front_anno, "FRONT")
        self.FRONT_LEFT = Camera(img_path, imgs[self.length:self.length*2], front_left_anno, "FRONT_LEFT")
        self.FRONT_RIGHT = Camera(img_path, imgs[self.length*2:self.length*3], front_right_anno, "FRONT_RIGHT")
        self.SIDE_LEFT = Camera(img_path, imgs[self.length*3:self.length*4], side_left_anno, "SIDE_LEFT")
        self.SIDE_RIGHT = Camera(img_path, imgs[self.length*4:], side_right_anno, "SIDE_RIGHT")

    def get_annotation(self):
        front_anno = []
        front_left_anno = []
        front_right_anno = []
        side_left_anno = []
        side_right_anno = []
        with open(self.ann_path, 'rb') as f:
            groundtruth = pickle.load(f)
        for i in range(len(groundtruth)):
            if i % 5 == 0:
                front_anno.append(groundtruth[i])
            elif i % 5 == 1:
                front_left_anno.append(groundtruth[i])
            elif i % 5 == 2:
                front_right_anno.append(groundtruth[i])
            elif i % 5 == 3:
                side_left_anno.append(groundtruth[i])
            elif i % 5 == 4:
                side_right_anno.append(groundtruth[i])
        return front_anno, front_left_anno, front_right_anno, side_left_anno, side_right_anno

    def __getitem__(self, idx):
        imgs = []
        targets = []
        img, target = self.FRONT.__getitem__(idx)
        img=cv2.undistort(img,self.intrinsic[0],self.disorted_coeff[0])
        imgs.append(img)
        targets.append(target)

        img, target = self.FRONT_LEFT.__getitem__(idx)
        img=cv2.undistort(img,self.intrinsic[1],self.disorted_coeff[1])
        imgs.append(img)
        targets.append(target)

        img, target = self.FRONT_RIGHT.__getitem__(idx)
        img=cv2.undistort(img,self.intrinsic[2],self.disorted_coeff[2])
        imgs.append(img)
        targets.append(target)

        img, target = self.SIDE_LEFT.__getitem__(idx)
        img=cv2.undistort(img,self.intrinsic[3],self.disorted_coeff[3])
        imgs.append(img)
        
        targets.append(target)

        img, target = self.SIDE_RIGHT.__getitem__(idx)
        img=cv2.undistort(img,self.intrinsic[4],self.disorted_coeff[4])
        imgs.append(img)
        targets.append(target)
        return imgs, targets
    
    def make_intrinsic_mat(self, param):
        # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
        intrinsics = []
        distCoffs = []
        for i in range(CAMMERA_NUM):
            intrinsic = np.array(
                [[param[i][0], 0, param[i][2]], [0, param[i][1], param[i][3]], [0, 0, 1]])
            # intrinsic=np.asmatrix(intrinsic)
            dist = np.array(param[i][4:])
            intrinsics.append(intrinsic)
            distCoffs.append(dist)
        return intrinsics, distCoffs

class Waymo3DLoader(torch.utils.data.Dataset):
    def __init__(self, root, segment):
        self.root = root
        self.sequence = segment
        self.anno = self.load_anno()
        self.pointcloudlist = list(sorted(os.listdir(root + segment)))[:-2]

    def load_anno(self):
        with open(self.root + self.sequence + "/" + self.sequence + ".pkl", 'rb') as f:
            groundtruth = pickle.load(f)
        return groundtruth

    def __getitem__(self, idx):
        target = self.anno[idx]
        pointcloud = np.load(self.root + self.sequence + "/" + self.pointcloudlist[idx])
        return pointcloud, target


if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequece = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    test = Waymo2DLoader(root, sequece)
    print(test.__getitem__(5))
    test1 = Waymo3DLoader(root, sequece)
