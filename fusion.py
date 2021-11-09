import numpy as np
import open3d as o3d
from modelmanager import ModelManager
import numpy as np
import cv2
import pickle
CAMMERA_NUM = 5


class Fusion(object):
    def __init__(self, root, ckpt):
        self.val =  ModelManager(root, ckpt)
        self.current_intrinsics = None
        self.current_distcoeff = None
        self.current_extrinsics = None

    def make_intrinsic_mat(self, param):
        # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
        intrinsics = []
        distCoffs = []
        for i in range(CAMMERA_NUM):
            intrinsic = np.array(
                [[param[i][0], 0, param[i][2], 0], [0, param[i][1], param[i][3], 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            # intrinsic=np.asmatrix(intrinsic)
            dist = np.array(param[i][4:])
            intrinsics.append(intrinsic)
            distCoffs.append(dist)
        return intrinsics, distCoffs

    def make_extrinsic_mat(self, param):
        extrinscis = []
        for i in range(CAMMERA_NUM):
            # extrinsic=np.asmatrix(param[i])
            extrinscis.append(param[i])
        return extrinscis

    def calibration(self):
        # annos3d, annos2d = self.val.val()
        with open("anno3d.pkl",'rb')as f:
            annos3d=pickle.load(f)
        with open("anno2d.pkl",'rb')as f:
            annos2d=pickle.load(f)
        sequence = annos2d[0]["frame_id"][0][0:-4]
        self.current_intrinsics=annos2d[0]["intrinsic"]
        self.current_extrinsics = self.make_extrinsic_mat(annos2d[0]["extrinsic"])
        result = []
        for i in range(len(annos2d)):
            img3d = {}
            if annos2d[i]["frame_id"][0][0:-4] != sequence:
                self.current_intrinsics=annos2d[i]["intrinsic"]
                self.current_extrinsics = self.make_extrinsic_mat(annos2d[i]["extrinsic"])
                annos2d[i]["frame_id"][0][0:-4] = sequence
            for cam_num, img in enumerate(annos2d[i]["imgs"]):

                img3d["frame_id"] = sequence
                img3d["filename"] = annos2d[i]["image_id"]   
        return result

    def visuallize(self):
        return NotImplemented


if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequence = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    fuse = Fusion(root, ckpt)
    fuse.calibration()
