import numpy as np
import open3d as o3d
from torch._C import device, dtype
from torch.autograd.grad_mode import F
from modelmanager import ModelManager
import numpy as np
import cv2
import torch
import pickle

CAMMERA_NUM = 5

class Fusion(object):
    def __init__(self, root, ckpt):
        self.val = ModelManager(root, ckpt)
        self.current_intrinsics = None
        self.current_distcoeff = None
        self.current_extrinsics = None
        self.root = root

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
        with open("anno3d.pkl", 'rb')as f:
            annos3d = pickle.load(f)
        with open("anno2d.pkl", 'rb')as f:
            annos2d = pickle.load(f)
        sequence = annos2d[0]["frame_id"][0][0:-4]
        self.current_intrinsics = annos2d[0]["intrinsic"]
        self.current_extrinsics = self.make_extrinsic_mat(
            annos2d[0]["extrinsic"])
        result = []
        for i in range(len(annos2d)):  # all sequence
            img3d = {}
            if annos2d[i]["frame_id"][0][0:-4] != sequence:
                self.current_intrinsics = annos2d[i]["intrinsic"]
                self.current_extrinsics = self.make_extrinsic_mat(
                    annos2d[i]["extrinsic"])
                annos2d[i]["frame_id"][0][0:-4] = sequence
            xyz = np.load(self.root+sequence+'/0' +
                          annos2d[i]["frame_id"][0][-3:]+".npy")[:, :3]
            point_planes = self.pointcloud2image(xyz)
            print("{0} ==> calibartion complete".format(
                sequence+'/0'+annos2d[i]["frame_id"][0][-3:]))
            frustrum_for_onescene = self.make_frustrum(
                annos2d[i]["anno"], xyz, point_planes)

            img3d["frustrum"]=frustrum_for_onescene
            img3d["frame_id"] = sequence
            img3d["filename"] = annos2d[i]["image_id"]
            result.append(img3d)
        with open("frustrum.pkl", 'wb') as f:
            pickle.dump(result, f)
        return result

    def pointcloud2image(self, lidar):
        point_planes = []
        one = np.ones((len(lidar), 1))
        cp_lidar = np.concatenate((lidar, one), axis=1)
        cp_lidar = torch.from_numpy(cp_lidar).cuda()
        for camera_num in range(3):
            to_plane = torch.matmul(torch.linalg.inv(torch.from_numpy(
                self.current_extrinsics[camera_num]).type(torch.float64).cuda()), cp_lidar.T).T
            change_coordinate = torch.stack(
                [-1*to_plane[:, 1], -1*to_plane[:, 2], to_plane[:, 0]], dim=0)
            to_image = torch.matmul(torch.from_numpy(self.current_intrinsics[camera_num]).type(
                torch.float64).cuda(), change_coordinate).T
            to_image = to_image/to_image[:, 2, None]
            point_image = self.point_to_tensor(to_image, 1920, 1280)
            point_planes.append(point_image.numpy())
            # print("calibaration complete camera number: {0}".format(camera_num))
        for camera_num in [3, 4]:
            point_image = torch.zeros(
                [1920, 886, 5], dtype=torch.int32, device='cuda:0')
            to_plane = torch.matmul(torch.linalg.inv(torch.from_numpy(
                self.current_extrinsics[camera_num]).type(torch.float64).cuda()), cp_lidar.T).T
            change_coordinate = torch.stack(
                [-1*to_plane[:, 1], -1*to_plane[:, 2], to_plane[:, 0]], dim=0)
            to_image = torch.matmul(torch.from_numpy(self.current_intrinsics[camera_num]).type(
                torch.float64).cuda(), change_coordinate).T
            to_image = to_image/to_image[:, 2, None]
            point_plane = self.point_to_tensor(to_image, 1920, 1280)
            point_planes.append(point_plane.numpy())
            # print("calibaration complete camera number: {0}".format(camera_num))
        return point_planes

    def point_to_tensor(self, calibrated_point, width, height):
        point_image = -1 * \
            torch.ones([width, height, 4], dtype=torch.int32, device='cuda:0')
        idx_tensor = torch.zeros([width, height], dtype=torch.int32)
        for idx, point in enumerate(calibrated_point):
            pixel_x = int(torch.floor(point[0]))
            pixel_y = int(torch.floor(point[1]))
            if pixel_x > 0 and pixel_x < width:
                if pixel_y > 0 and pixel_y < height:
                    point_image[pixel_x][pixel_y][idx_tensor[pixel_x]
                                                  [pixel_y]] = idx
                    # print(point_image[pixel_x][pixel_y][idx_tensor[pixel_x][pixel_y]])
                    # print(idx_tensor[pixel_x][pixel_y])
                    idx_tensor[pixel_x][pixel_y] += 1
                    # print(idx_tensor[pixel_x][pixel_y])
        return point_image.cpu()

    def make_frustrum(self, annos, xyz, point_planes):
        frustrums=[]
        for camera_num in range(CAMMERA_NUM):
            for i, label in enumerate(annos[camera_num]["labels"]):
                idx=None
                if label != "unknown":
                    projected_point = {}
                    projected_point["label"] = label
                    box=[annos[camera_num]["boxes"][i][0],annos[camera_num]["boxes"][i][1],annos[camera_num]["boxes"][i][2],annos[camera_num]["boxes"][i][3]]
                    box=np.floor(box).astype(np.int)
                    # print(point_planes[camera_num][int((box[0]+box[1])/2)][(int(box[0]+box[1])/2)])
                    frustrum=np.unique(point_planes[camera_num][box[0]:box[1],box[2]:box[3]].flatten("C"))
                    idx=np.where(frustrum==-1)

                    if idx is not None:
                        frustrum=np.delete(frustrum,idx)
                    projected_point["frustrum"]=frustrum
                    frustrums.append(projected_point)
        return frustrums
    # def find_mid_point(self,box,point_plane):
    #     mid=[(box[0]+box[1])/2,(box[2]+box[3])/2]
    #     point_plane[(box[0]+box[1])/2][(box[0]+box[1])/2]
    # def projection(self,annos,lidar):

    #     frustrum_result=[]
    #     one=np.ones((len(lidar),1))
    #     cp_lidar=np.concatenate((lidar,one),axis=1)
    #     # p=[]
    #     # Q=Queue()
    #     for camera_num, anno in enumerate(annos):
    #         to_plane=np.matmul(np.linalg.inv(self.current_extrinsics[camera_num])[0:3,:],cp_lidar.T).T  #포인트를 카메라좌표계로 변환
    #     #     procsess=Process(target=self.make_frustrum,args=(camera_num,anno,to_plane,Q,))
    #     #     p.append(procsess)
    #     # for i in range(5):
    #     #     p[i].start()

    #     # for i in range(5):
    #     #     p[i].join()
    #     # for i in range(5):
    #     #     frustrum_result.append(Q.get())
    #     #     p[i].close()
    #         for i , label in enumerate(anno["labels"]):
    #             if label !="unknown":
    #                 projected_point={}
    #                 projected_point["label"]=label
    #                 camera=viewbox(anno["boxes"][i])
    #                 physical_plane=np.matmul(np.linalg.inv(self.current_intrinsics[camera_num]),camera.boxmat).T #각박스의 피지컬 프레인을 구함
    #                 to_plane=np.concatenate(([-1*to_plane[:,1].T],[-1*to_plane[:,2].T],[to_plane[:,0].T]),axis=0).T
    #                 toplaneidx=[]
    #                 for i,point in enumerate(to_plane):
    #                     if point[2]>0:
    #                         toplaneidx.append(i)
    #                 normal=to_plane
    #                 normal=normal/normal[:,2][:,None]
    #                 idx=[]
    #                 for i in toplaneidx:
    #                     if normal[i,0]>physical_plane[0,0]:
    #                         if normal[i,1]>physical_plane[0,1]:
    #                             if normal[i,0]<physical_plane[1,0]:
    #                                 if normal[i,1]<physical_plane[1,1]:
    #                                     idx.append(i)
    #                 projected_point["frustrum"]=idx #각 이미지박스 별로 프러스트럼만들어지면 그포인트 들의 인덱스를 리스트에 저장
    #             frustrum_result.append(projected_point)
    #     return frustrum_result

    def visuallize(self):
        return NotImplemented


if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequence = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    fuse = Fusion(root, ckpt)
    fuse.calibration()
