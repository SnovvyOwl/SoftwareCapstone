from validiation import Validation
import numpy as np
import cv2
CAMMERA_NUM=5
class Fusion(object):
    def __init__(self,root,ckpt):
        self.val=Validation(root,ckpt)
        self.current_intrinsics=None
        self.current_distcoeff=None
        self.current_extrinsics=None

    def make_intrinsic_mat(self,param):
        # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
        intrinsics=[]
        distCoffs=[]
        for i in range(CAMMERA_NUM):
            intrinsic=np.array([[param[i][0],0,param[i][2]],[0, param[i][1], param[i][3]],[0,0,1]])
            intrinsic=np.asmatrix(intrinsic)
            dist=np.array(param[i][4:])
            intrinsics.append(intrinsic)
            distCoffs.append(dist)
        return intrinsics,distCoffs
    
    def make_extrinsic_mat(self,param):
        extrinscis=[]
        for i in range(CAMMERA_NUM):
            extrinsic=np.asmatrix(param[i])
            extrinscis.append(extrinsic)
        return extrinscis

    def calibration(self):
        annos3d,annos2d=self.val.val()
        sequence=annos2d[0]["frame_id"][0][0:-4]
        self.current_intrinsics,self.current_distcoeff=self.make_intrinsic_mat(annos2d[0]["intrinsic"])
        self.current_extrinsics=self.make_extrinsic_mat(annos2d[0]["extrinsic"])
        for i in range(len(annos2d)):
            if annos2d[i]["frame_id"][0][0:-4] != sequence:
                self.current_intrinsics,self.current_distcoeff=self.make_intrinsic_mat(annos2d[i]["intrinsic"])
                self.current_extrinsics=self.make_extrinsic_mat(annos2d[i]["extrinsic"])
            # for cam_num, img in enumerate(annos2d[i]["imgs"]):
            #     img_udist=cv2.undistort(np.array(img),self.current_intrinsics[cam_num],self.current_distcoeff[cam_num]) 
            img=np.array(annos2d[i]["imgs"][0])
            img=img[:, :, ::-1].copy()
            img_udist=cv2.undistort(img,self.current_intrinsics[0],self.current_distcoeff[0])
            cv2.imwrite(filename="before.jpg",img=img)
            cv2.imwrite(filename="after.jpg",img=img_udist)    
           

    def visuallize(self):
        return NotImplemented
 


if __name__ == "__main__":
    root="./data/waymo/waymo_processed_data/"
    sequece='segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt="./checkpoints/checkpoint_epoch_30.pth"
    fuse=Fusion(root,ckpt)
    fuse.calibration()