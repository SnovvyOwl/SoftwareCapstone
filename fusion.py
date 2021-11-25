from operator import pos
from re import T
import numpy as np
import open3d as o3d
from modelmanager import ModelManager
import numpy as np
import torch
import pickle
from queue import Queue
import multiprocessing as mp
from tqdm import tqdm
CAMMERA_NUM = 5


def make_3dBox(anno):
    boxes = []
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [4, 5],
             [5, 6], [6, 7], [4, 7], [1, 5], [2, 6], [3, 7]]
    corners3d = boxes_to_corners_3d(anno["boxes_lidar"])
    for box in corners3d:
        box3d = o3d.geometry.LineSet()
        box3d.points = o3d.utility.Vector3dVector(box)
        box3d.lines = o3d.utility.Vector2iVector(lines)
        boxes.append(box3d)
    return boxes


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


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
    
    def set_matrix(self):
        with open("anno3d.pkl", 'rb')as f:
            annos3d = pickle.load(f)
        with open("anno2d.pkl", 'rb')as f:
            annos2d = pickle.load(f)
        self.current_intrinsics = annos2d[0]["intrinsic"]
        self.current_extrinsics = self.make_extrinsic_mat(annos2d[0]["extrinsic"])
    
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
        for i in tqdm((range(len(annos2d)))):  # all sequence
            img3d = {}
            if annos2d[i]["frame_id"][0][0:-4] != sequence:
                self.current_intrinsics = annos2d[i]["intrinsic"]
                self.current_extrinsics = self.make_extrinsic_mat(annos2d[i]["extrinsic"])
                annos2d[i]["frame_id"][0][0:-4] = sequence
            xyz = np.load(self.root+sequence+'/0'+annos2d[i]["frame_id"][0][-3:]+".npy")[:, :3]
            point_planes = self.pointcloud2image(xyz)
            # print("{0} ==> calibartion complete".format(
                # sequence+'/0'+annos2d[i]["frame_id"][0][-3:]))
            frustrum_for_onescene = self.make_frustrum(annos2d[i]["anno"], xyz, point_planes)
            # seg_result = self.segmetation(xyz, frustrum_for_onescene)
            img3d["frustrum"] = frustrum_for_onescene
            img3d["frame_id"] = sequence
            img3d["filename"] = annos2d[i]["image_id"]
            # img3d["seg"]=seg_result
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
            to_plane = torch.matmul(torch.linalg.inv(torch.from_numpy(self.current_extrinsics[camera_num]).type(torch.float64).cuda()), cp_lidar.T).T
            change_coordinate = torch.stack([-1*to_plane[:, 1], -1*to_plane[:, 2], to_plane[:, 0]], dim=0).cpu().numpy().T
            # img=[]
            # for point in change_coordinate:
            #     if point[2]>0:
            #         img.append(point)
            lidar_idx=np.array((range(len(lidar))))
            negative_idx=np.where(change_coordinate[:,2]<=0)
            change_coordinate=np.delete(change_coordinate,negative_idx,axis=0).T
            lidar_idx=np.delete(lidar_idx,negative_idx)
            change_coordinate=torch.from_numpy(change_coordinate).type(torch.float64).cuda()
            to_image = torch.matmul(torch.from_numpy(self.current_intrinsics[camera_num]).type(torch.float64).cuda(), change_coordinate).T
            to_image = to_image/to_image[:, 2, None]
            point_plane = self.point_to_tensor(to_image,lidar_idx, 1920, 1280)
            point_planes.append(point_plane.numpy())
            # print("calibaration complete camera number: {0}".format(camera_num))
        for camera_num in [3, 4]:
            to_plane = torch.matmul(torch.linalg.inv(torch.from_numpy(self.current_extrinsics[camera_num]).type(torch.float64).cuda()), cp_lidar.T).T
            change_coordinate = torch.stack([-1*to_plane[:, 1], -1*to_plane[:, 2], to_plane[:, 0]], dim=0).cpu().numpy().T
            lidar_idx=np.array((range(len(lidar))))
            negative_idx=np.where(change_coordinate[:,2]<=0)
            change_coordinate=np.delete(change_coordinate,negative_idx,axis=0).T
            lidar_idx=np.delete(lidar_idx,negative_idx)
            change_coordinate=torch.from_numpy(change_coordinate).type(torch.float64).cuda()
            to_image = torch.matmul(torch.from_numpy(self.current_intrinsics[camera_num]).type(torch.float64).cuda(), change_coordinate).T
            to_image = to_image/to_image[:, 2, None]
            point_plane = self.point_to_tensor(to_image, lidar_idx,1920, 886)
            point_planes.append(point_plane.numpy())
            # print("calibaration complete camera number: {0}".format(camera_num))
        return point_planes

    def point_to_tensor(self, calibrated_point, lidar_idx,width, height):
        point_image = -1 * torch.ones([width, height, 4], dtype=torch.int32, device='cuda:0')
        idx_tensor = torch.zeros([width, height], dtype=torch.int32)
        tmp=[]
        for idx, point in enumerate(calibrated_point):
            pixel_x = int(torch.floor(point[0]))
            pixel_y = int(torch.floor(point[1]))
            if pixel_x > 0 and pixel_x < width:
                if pixel_y > 0 and pixel_y < height:
                    point_image[pixel_x][pixel_y][idx_tensor[pixel_x][pixel_y]] =lidar_idx[idx]
                    # print(point_image[pixel_x][pixel_y][idx_tensor[pixel_x][pixel_y]])
                    # print(idx_tensor[pixel_x][pixel_y])
                    idx_tensor[pixel_x][pixel_y] += 1
                    # print(idx_tensor[pixel_x][pixel_y])
                    tmp.append(idx)
        # np.savetxt("fusion.txt",np.array(tmp),fmt='%6.6e')
        return point_image.cpu()
    
    def doitwell(self,planes):
        frustrums=[]
        for plane in planes:
            frustrum=np.unique(plane.flatten("C"))
            idx = np.where(frustrum == -1)
            if idx is not None:
                frustrum = np.delete(frustrum, idx)
            frustrums.append(frustrum)
    
        return frustrums
    def make_frustrum(self, annos, xyz, point_planes):
        frustrums = []
        for camera_num in range(CAMMERA_NUM):
            for i, label in enumerate(annos[camera_num]["labels"]):
                idx = None
                if label != "unknown":
                    projected_point = {}
                    projected_point["label"] = label
                    box = [annos[camera_num]["boxes"][i][0], annos[camera_num]["boxes"][i][2], annos[camera_num]["boxes"][i][1], annos[camera_num]["boxes"][i][3]]
                    box = np.floor(box).astype(np.int)
                    # print(point_planes[camera_num][int((box[0]+box[1])/2)][(int(box[0]+box[1])/2)])
                    frustrum = np.unique(point_planes[camera_num][box[0]:box[1], box[2]:box[3]].flatten("C"))
                    idx = np.where(frustrum == -1)

                    if idx is not None:
                        frustrum = np.delete(frustrum, idx)
                    projected_point["frustrum"] = frustrum
                    
                    if frustrum.size != 0:
                        # self.find_centroid2(point_planes[camera_num],box)
                        centroid, centorid_idx, frustrum_idx = self.find_centroid(xyz[frustrum], frustrum)
                        projected_point["centroid"] = centroid
                        projected_point["centroid_idx"] = centorid_idx
                        projected_point["frustrum_idx"]=frustrum_idx
                    else:
                        projected_point["centroid"] = None
                        projected_point["centroid_idx"] = None
                        # projected_point["frustrum_idx"]=None
                    frustrums.append(projected_point)
        return frustrums
    def find_centroid2(self,plane,box):
        center_pixel=10
        center_plane=np.unique(plane[int((box[0]+box[1])/2)-center_pixel:int((box[0]+box[1])/2)+center_pixel, int((box[2]+box[3])/2)-center_pixel:int((box[2]+box[3])/2)+center_pixel])
       
        print(center_plane)
        # return centroid,centroid_idx
    def find_centroid(self, frustrum, frustrum_idx):
        min_radius = (frustrum[:, 0]**2+frustrum[:, 1]** 2+frustrum[:, 2]**2)**0.5
        min_radius = min_radius[np.argmin(min_radius)]
        mean_radius = (frustrum[:, 0].mean()**2+frustrum[:,1].mean()**2+frustrum[:, 2].mean()**2)**0.5
        centroid_vector = [frustrum[:, 0].mean()/mean_radius*min_radius, frustrum[:, 1].mean()/mean_radius*min_radius, (frustrum[:, 2]).mean()/mean_radius*min_radius]
        centroid = None
        radius = (frustrum[:, 1]-centroid_vector[1])**2 + (frustrum[:, 2]-centroid_vector[2])**2
        centroid = frustrum[np.argmin(radius)]
        centroid_idx = frustrum_idx[np.argmin(radius)]
        return centroid, centroid_idx, np.argmin(radius)

    def segmetation(self, all_point, frustrums):
        seg_res=[]
        que=mp.Queue()
        tmp=[]
        for f in frustrums:
            if f["centroid_idx"] is not None:
                tmp.append(f)
        start_pos=0
        div=mp.cpu_count()
        end_pos=len(frustrums)
        for i in tqdm(range(start_pos, end_pos + div, div)):
            current=tmp[start_pos:start_pos + div]
            self.make_cluster(current[0],all_point,que,0.007)
            res={}
            if current!=[]:
                procs=[]
                for frustrum in current:
                    proc=mp.Process(target=self.make_cluster,args=(frustrum,all_point,que,0.05))
                    procs.append(proc)
                    proc.start()
                for proc in procs:
                    seg_res.append(que.get())
                    proc.join()
                for proc in procs:
                    proc.close()
            start_pos = start_pos + div
        return seg_res

    def make_cluster(self,frustrum, all_point,que,max_radius=0.01):
        res={}
        res["label"]=frustrum["label"]
        res["centroid"]=frustrum["centroid"]
        res["centroid_idx"]=frustrum["centroid_idx"]
        cluster = []
        centroid=frustrum["centroid"]
        centroid_idx=frustrum["centroid_idx"]
        points = Queue()
        idices = Queue()
        points.put(centroid)
        idx=list(range(all_point.shape[0]))
        idices.put(idx[centroid_idx])
        cluster.append(idx[centroid_idx])
        cp_all_point = all_point.copy()
        cp_all_point = np.delete(cp_all_point, centroid_idx, 0)
        idx = np.delete(idx, centroid_idx)
        while points.empty() != True:
            leaf = points.get()
            radius = np.array((cp_all_point[:, 0]-leaf[0])**2+(cp_all_point[:, 1]-leaf[1])**2+(cp_all_point[:, 2]-leaf[2])**2)
            cluster_idx = np.where(radius < max_radius)
            cluster.extend(idx[cluster_idx[0][:]])
            next_idx = np.delete(idx, cluster_idx[0])
            next_cp_all_point = np.delete(cp_all_point, cluster_idx[0], 0)
            for i in list(cluster_idx[0][:]):
                points.put(cp_all_point[i])
            idx = next_idx
            cp_all_point = next_cp_all_point   
        que.put(cluster)


if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequence = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    fuse = Fusion(root, ckpt)
    fuse.calibration()
