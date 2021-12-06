import pickle
import numpy as np
import torch
import open3d as o3d
from fusion import Fusion

class Inference(object):
    def __init__(self,root,ckptdir):
        self.root=root
        self.ckptdir=ckptdir
        self.fusion=Fusion(root,ckptdir)
    
    def main(self):
        # result_of_fusion= self.fusion.main()
        with open("frustum.pkl",'rb')as f:
            result_of_fusion=pickle.load(f)

if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    infer=Inference(root,ckpt)
    infer.main()
