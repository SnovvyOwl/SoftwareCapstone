import numpy as np
import open3d as o3d
from modelmanager import ModelManager
import numpy as np
import cv2
import pickle
CAMMERA_NUM = 5

class viewbox(object):
    def __init__(self,boxes):
       
        self.min=np.array([[float(boxes[0])],[float(boxes[1])],[1]])
        self.max=np.array([[float(boxes[3])],[float(boxes[3])],[1]])
        self.boxmat=np.concatenate((self.min,self.max) ,axis=1)

class Fusion(object):
    def __init__(self, root, ckpt):
        self.val =  ModelManager(root, ckpt)
        self.current_intrinsics = None
        self.current_distcoeff = None
        self.current_extrinsics = None
        self.root=root
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
        for i in range(len(annos2d)): # all sequence
            img3d = {}
            if annos2d[i]["frame_id"][0][0:-4] != sequence: 
                self.current_intrinsics=annos2d[i]["intrinsic"]
                self.current_extrinsics = self.make_extrinsic_mat(annos2d[i]["extrinsic"])
                annos2d[i]["frame_id"][0][0:-4] = sequence
            xyz=np.load(self.root+sequence+'/0'+annos2d[i]["frame_id"][0][-3:]+".npy")[:,:3]
            frustrum_for_onescene=self.projection(annos2d[i]["anno"],xyz)
            img3d["frame_id"] = sequence
            img3d["filename"] = annos2d[i]["image_id"]   
        return result

    def projection(self,annos,lidar):
        frustrum_result=[]
        for camera_num, anno in enumerate(annos):
            for i , label in enumerate(anno["labels"]):
                if label !="unknown":
                    projected_point={}
                    projected_point["label"]=label
                    camera=viewbox(anno["boxes"][i])
                    physical_plane=np.matmul(np.linalg.inv(self.current_intrinsics[camera_num]),camera.boxmat).T #각박스의 피지컬 프레인을 구함
                    one=np.ones((len(lidar),1))
                    cp_lidar=np.concatenate((lidar,one),axis=1)
                    to_plane=np.matmul(np.linalg.inv(self.current_extrinsics[camera_num])[0:3,:],cp_lidar.T).T  #포인트를 카메라좌표계로 변환
                    to_plane=np.concatenate(([-1*to_plane[:,1].T],[-1*to_plane[:,2].T],[to_plane[:,0].T]),axis=0).T
                    toplaneidx=[]
                    for i,point in enumerate(to_plane):
                        if point[2]>0:
                            toplaneidx.append(i)
                    normal=to_plane
                    normal=normal/normal[:,2][:,None]
                    idx=[]
                    for i in toplaneidx:
                        if normal[i,0]>physical_plane[0,0]:
                            if normal[i,1]>physical_plane[0,1]:
                                if normal[i,0]<physical_plane[1,0]:
                                    if normal[i,1]<physical_plane[1,1]:
                                        idx.append(i)
                    projected_point["frustrum"]=idx #각 이미지박스 별로 프러스트럼만들어지면 그포인트 들의 인덱스를 리스트에 저장
                frustrum_result.append(projected_point)
        return frustrum_result
    
    def visuallize(self):
        return NotImplemented


if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequence = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    fuse = Fusion(root, ckpt)
    fuse.calibration()
