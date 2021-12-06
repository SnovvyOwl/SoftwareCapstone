import pickle
import numpy as np
import torch
import open3d as o3d
from fusion import Fusion
from PVRCNN.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
class Inference(object):
    def __init__(self,root,ckptdir):
        self.root=root
        self.ckptdir=ckptdir
        self.current_segment=None
        # self.fusion=Fusion(root,ckptdir)
    
    def main(self):
        # result_of_fusion,annos3d= self.fusion.main()
        with open("frustum.pkl",'rb')as f:
            result_of_fusion=pickle.load(f)
        with open("anno3d.pkl", 'rb')as f:
            annos3d = pickle.load(f)
        for result in result_of_fusion:
            print(result)
    def load_ground_truth_data(self):
        print("gt")

if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    infer=Inference(root,ckpt)
    infer.main()
    dir="/home/seongwon/SoftwareCapstone/data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/segment-1024360143612057520_3580_000_3600_000_with_camera_labels.pkl"
