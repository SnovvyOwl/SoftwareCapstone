from operator import pos
from re import I, T
import numpy as np
from numpy.core.numeric import False_
import open3d as o3d
from modelmanager import ModelManager
import numpy as np
import torch
import pickle
from queue import Queue
import multiprocessing as mp
from tqdm import tqdm
import PVRCNN.utils.common_utils
import PVRCNN.ops.roiaware_pool3d.roiaware_pool3d_utils
CAMMERA_NUM = 5

def remove_points_in_boxes3d(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    boxes3d, is_numpy = PVRCNN.utils.common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = PVRCNN.utils.common_utils.check_numpy_to_torch(points)
    point_masks =  PVRCNN.ops.roiaware_pool3d.roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points

def iou2d(box1,box2):
    # input  : box1[4], box2[4]
    # output : IoU
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] ) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0] ) * (box2[3] - box2[1])

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

# def make_3dBox(anno):
#     boxes = []
#     lines = [[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [4, 5],
#              [5, 6], [6, 7], [4, 7], [1, 5], [2, 6], [3, 7]]
#     corners3d = boxes_to_corners_3d(anno["boxes_lidar"])
#     for box in corners3d:
#         box3d = o3d.geometry.LineSet()
#         box3d.points = o3d.utility.Vector3dVector(box)
#         box3d.lines = o3d.utility.Vector2iVector(lines)
#         boxes.append(box3d)
#     return boxes


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
        self.current_extrinsics = self.make_extrinsic_mat(annos2d[0]["extrinsic"])
        result = []
        for i in tqdm((range(len(annos2d)))):  # all sequence
            img3d = {}
            if annos2d[i]["frame_id"][0][0:-4] != sequence:
                self.current_intrinsics = annos2d[i]["intrinsic"]
                self.current_extrinsics = self.make_extrinsic_mat(annos2d[i]["extrinsic"])
                annos2d[i]["frame_id"][0][0:-4] = sequence
            xyz = np.load(self.root+sequence+'/0'+annos2d[i]["frame_id"][0][-3:]+".npy")[:, :3]
            point_planes = self.pointcloud2image(xyz)
            # print("{0} ==> calibartion complete".format(sequence+'/0'+annos2d[i]["frame_id"][0][-3:]))
            frustum_for_onescene = self.make_frustum(annos2d[i]["anno"], xyz, point_planes)
            box3d_to_2d=self.box_is_in_plane(annos3d[i])
            # seg_result = self.segmetation(xyz, frustum_for_onescene)
            res=self.is_box_in_frustum(frustum_for_onescene,box3d_to_2d,xyz)
            # res=self.is_3d_box_frustum(annos3d[i]["box_lidars"],frustum_for_onescene)
            img3d["frustum"] = res
            img3d["frame_id"] = sequence
            img3d["filename"] = annos2d[i]["image_id"]
            # img3d["seg"]=seg_result
            result.append(img3d)
        with open("frustum.pkl", 'wb') as f:
            pickle.dump(result, f)
        return result

    def is_3d_box_in_frustum(self,boxes,frustum_for_onescene):
        return NotImplementedError
    
    def pointcloud2image(self, lidar):
        #input Lidar PonitCloud
        #return lidar point plane
        point_planes = []
        one = np.ones((len(lidar), 1))
        cp_lidar = np.concatenate((lidar, one), axis=1)
        cp_lidar = torch.from_numpy(cp_lidar).cuda()

        for camera_num in range(3):
            to_plane = torch.matmul(torch.linalg.inv(torch.from_numpy(self.current_extrinsics[camera_num]).type(torch.float64).cuda()), cp_lidar.T).T
            change_coordinate = torch.stack([-1*to_plane[:, 1], -1*to_plane[:, 2], to_plane[:, 0]], dim=0).cpu().numpy().T
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
        for idx, point in enumerate(calibrated_point):
            pixel_x = int(torch.floor(point[0]))
            pixel_y = int(torch.floor(point[1]))
            if pixel_x > 0 and pixel_x < width:
                if pixel_y > 0 and pixel_y < height:
                    point_image[pixel_x][pixel_y][idx_tensor[pixel_x][pixel_y]] =lidar_idx[idx]
                    # print(point_image[pixel_x][pixel_y][idx_tensor[pixel_x][pixel_y]])
                    # print(idx_tensor[pixel_x][pixel_y])
                    idx_tensor[pixel_x][pixel_y] += 1
        return point_image.cpu()
 

    def make_frustum(self, annos, xyz, point_planes):
        frustums = []
        for camera_num in range(CAMMERA_NUM):
            for i, label in enumerate(annos[camera_num]["labels"]):
                idx = None
                if label != "unknown":
                    if annos[camera_num]['scores'][i]>0.3:
                        projected_point = {}
                        projected_point["label"] = label
                        #box = [annos[camera_num]["boxes"][i][0], annos[camera_num]["boxes"][i][2], annos[camera_num]["boxes"][i][1], annos[camera_num]["boxes"][i][3]]
                        box=annos[camera_num]["boxes"][i]
                        box = np.floor(box).astype(np.int)
                        # print(point_planes[camera_num][int((box[0]+box[1])/2)][(int(box[0]+box[1])/2)])
                        frustum = np.unique(point_planes[camera_num][box[0]:box[2], box[1]:box[3]].flatten("C"))
                        idx = np.where(frustum == -1)
                        if idx is not None:
                            frustum = np.delete(frustum, idx)
                        # make 10% Large Frustum
                        x_extend=((box[2]-box[0])*1.1)/2
                        x_center=(box[2]+box[0])/2
                        y_extend=((box[3]-box[1])*1.1)/2
                        y_center=(box[3]+box[1])/2
                        large_box=[x_center-x_extend,y_center-y_extend,x_center+x_extend,y_center+y_extend]
                        large_box = np.floor(large_box).astype(np.int)
                        large_frustum = np.unique(point_planes[camera_num][large_box[0]:large_box[2], large_box[1]:large_box[3]].flatten("C"))
                        idx = np.where(large_frustum  == -1)
                        if idx is not None:
                            large_frustum = np.delete(large_frustum , idx)
                       
                        projected_point["large_frustum"]=large_frustum
                        projected_point["frustum"] = frustum
                        projected_point["2d_box"]=box
                        if frustum.size != 0:
                             # make center box
                            x_center_extend=((box[2]-box[0])*0.05)/2
                            y_center_extend=((box[3]-box[1])*0.05)/2
                            center_box=[x_center-x_center_extend,y_center-y_center_extend,x_center+x_center_extend,y_center+y_center_extend]
                            center_box=np.floor(center_box).astype(np.int)
                            center_frustum = np.unique(point_planes[camera_num][center_box[0]:center_box[2], center_box[1]:center_box[3]].flatten("C"))
                            idx = np.where(center_frustum  == -1)
                            if idx is not None:
                                center_frustum = np.delete(center_frustum , idx)
                         
                            # centroid, centorid_idx, frustum_idx = self.find_centroid(xyz[frustum], frustum)
                            projected_point["centroid"] = xyz[center_frustum]
                            projected_point["centroid_idx"] = center_frustum
                            # projected_point["centroid"] = centroid
                            # projected_point["centroid_idx"] = centorid_idx
                            # projected_point["frustum_idx"]=frustum_idx
                        else:
                            projected_point["centroid"] = None
                            projected_point["centroid_idx"] = None
                            # projected_point["frustum_idx"]=None    
                        
                        frustums.append(projected_point)
        return frustums
 
   
    def box_is_in_plane(self,annos):
        #  3D box ->2D Image
        # INPUT PV-RCNN 3D Boxes Oneframe
        # Return camera_num, pixel
        cp_box=annos["boxes_lidar"].copy()
        box_center=cp_box[:,:3].copy()
        box_plane=self.pointcloud2image(box_center)
        res=[]
        for camera_num in range(CAMMERA_NUM):
            plane=[]
            maked_2dbox_with_label={}
            box_idx=np.unique(box_plane[camera_num].flatten("C"))
            idx = np.where(box_idx == -1)
            if idx is not None:
                box_idx = np.delete(box_idx, idx)
            corners_3d=boxes_to_corners_3d(cp_box[box_idx,:])
            for i,box in enumerate(corners_3d):
                box2d=self.make_2d_box(camera_num,box)
                maked_2dbox_with_label["3d_box"]=box
                maked_2dbox_with_label["box"]=box2d
                maked_2dbox_with_label["label"]=annos["name"][box_idx[i]]
                plane.append(maked_2dbox_with_label.copy())
        
            res.append(plane)
        return res
    
    def make_2d_box(self,camera_num,box):
        cp_box=box.copy()
        one=np.ones((len(box),1))
        cp_box=np.concatenate((cp_box,one),axis=1)
        to_plane = np.matmul(np.linalg.inv(self.current_extrinsics[camera_num]), cp_box.T).T
        change_coordinate =np.concatenate(([-1*to_plane[:,1].T],[-1*to_plane[:,2].T],[to_plane[:,0].T]),axis=0)
        to_image = np.matmul(self.current_intrinsics[camera_num],change_coordinate).T
        to_image = to_image/to_image[:, 2, None]
        box=[np.min(to_image[:,0]),np.min(to_image[:,1]),np.max(to_image[:,0]),np.max(to_image[:,1])]
        return box

    def is_box_in_frustum(self,frustum_per_onescene,boxes,xyz):
        cp_xyz=xyz.copy()
        # rm_xyz=remove_points_in_boxes3d(cp_xyz,boxes)
        for i,frustum in enumerate(frustum_per_onescene):
            found=False
            for box_in_camera_num in boxes:
                for  box in box_in_camera_num:
                    if frustum["label"]==box["label"]:
                        iou=iou2d(box["box"],frustum["2d_box"])
                        if iou>0.5:
                        
                            # print(iou)
                            frustum_per_onescene[i]["3d_box"]=box["3d_box"]
                            frustum_per_onescene[i]["is_generated"]=False
                            found=True
                            break
            if found is False:
                if frustum["centroid"] is not None:
                    frustum_per_onescene[i]["is_generated"]=True
                    frustum_per_onescene[i]["seg"]=self.make_3d_box(xyz[frustum["large_frustum"]],frustum["centroid"],frustum["large_frustum"],frustum["centroid_idx"])
                else:
                    frustum_per_onescene[i]["is_generated"]=False
                    frustum_per_onescene[i]["seg"]=None
        return frustum_per_onescene

    def make_3d_box(self,frustum_point,centroid_point,frustum_idx,centroid_idx):
        """
        Returns:
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        Args:
            boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        """
        seg_cluster,seg_idx=self.segmentation(frustum_point,centroid_point,frustum_idx,centroid_idx,max_radius=0.03)
        return seg_idx

    def segmentation(self,frustum_point,centroid_point,frustum_idx,centroid_idx,max_radius=0.03):
        points = Queue()
        cp_frustum_point=frustum_point.copy()
        cp_frustum_idx=frustum_idx.copy()
        cluster=[]
        for center_point in centroid_point:
            points.put(center_point)
            cluster.append(center_point)
        idx=[]
        while points.empty() != True:
            leaf = points.get()
            radius = np.array((cp_frustum_point[:, 0]-leaf[0])**2+(cp_frustum_point[:, 1]-leaf[1])**2+(cp_frustum_point[:, 2]-leaf[2])**2)
            cluster_idx = np.where(radius < max_radius)
            idx.extend(cp_frustum_idx[cluster_idx])
            next_cp_frustum_idx = np.delete(cp_frustum_idx, cluster_idx[0])
            next_cp_frustum_point = np.delete(cp_frustum_point, cluster_idx[0], 0)
            for i in list(cluster_idx[0][:]):
                cluster.append(cp_frustum_point[i])
                points.put(cp_frustum_point[i])
            cp_frustum_idx = next_cp_frustum_idx
            cp_frustum_point = next_cp_frustum_point
        return cluster, idx

    def find_centroid(self, frustum, frustum_idx):
        # function: Find frustum Centroid
        # input: frustum(x,y,z),frustum_idx (n)-> points idx in this frame
        # output: centorid(x,y,z),centroid_idx(n)-> points idx in this frame ,frustum inner index
        min_radius = (frustum[:, 0]**2+frustum[:, 1]** 2+frustum[:, 2]**2)**0.5
        min_radius = min_radius[np.argmin(min_radius)]
        mean_radius = (frustum[:, 0].mean()**2+frustum[:,1].mean()**2+frustum[:, 2].mean()**2)**0.5
        centroid_vector = [frustum[:, 0].mean()/mean_radius*min_radius, frustum[:, 1].mean()/mean_radius*min_radius, (frustum[:, 2]).mean()/mean_radius*min_radius]
        centroid = None
        radius = (frustum[:, 1]-centroid_vector[1])**2 + (frustum[:, 2]-centroid_vector[2])**2
        centroid = frustum[np.argmin(radius)]
        centroid_idx = frustum_idx[np.argmin(radius)]
        return centroid, centroid_idx, np.argmin(radius)
    
    
    #Test Code
    def set_matrix(self):
        with open("anno2d.pkl", 'rb')as f:
            annos2d = pickle.load(f)
        self.current_intrinsics = annos2d[0]["intrinsic"]
        self.current_extrinsics = self.make_extrinsic_mat(annos2d[0]["extrinsic"])
        
    def doitwell(self,planes):
        frustums=[]
        for plane in planes:
            frustum=np.unique(plane.flatten("C"))
            idx = np.where(frustum == -1)
            if idx is not None:
                frustum = np.delete(frustum, idx)
            frustums.append(frustum)
        return frustums
    
    # def segmetation(self, all_point, frustums):
    #     seg_res=[]
    #     que=mp.Queue()
    #     tmp=[]
    #     for f in frustums:
    #         if f["centroid_idx"] is not None:
    #             tmp.append(f)
    #     start_pos=0
    #     div=mp.cpu_count()
    #     end_pos=len(frustums)
    #     for i in range(start_pos, end_pos + div, div):
    #         current=tmp[start_pos:start_pos + div]
    #         self.make_cluster(current[0],all_point,que,0.007)
    #         res={}
    #         if current!=[]:
    #             procs=[]
    #             for frustum in current:
    #                 proc=mp.Process(target=self.make_cluster,args=(frustum,all_point,que,0.05))
    #                 procs.append(proc)
    #                 proc.start()
    #             for proc in procs:
    #                 seg_res.append(que.get())
    #                 proc.join()
    #             for proc in procs:
    #                 proc.close()
    #         start_pos = start_pos + div
    #     return seg_res

    # def make_cluster(self,frustum, all_point,que,max_radius=0.01):
    #     res={}
    #     res["label"]=frustum["label"]
    #     res["centroid"]=frustum["centroid"]
    #     res["centroid_idx"]=frustum["centroid_idx"]
    #     cluster = []
    #     centroid=frustum["centroid"]
    #     centroid_idx=frustum["centroid_idx"]
    #     points = Queue()
    #     idices = Queue()
    #     points.put(centroid)
    #     idx=list(range(all_point.shape[0]))
    #     idices.put(idx[centroid_idx])
    #     cluster.append(idx[centroid_idx])
    #     cp_all_point = all_point.copy()
    #     cp_all_point = np.delete(cp_all_point, centroid_idx, 0)
    #     idx = np.delete(idx, centroid_idx)
    #     while points.empty() != True:
    #         leaf = points.get()
    #         radius = np.array((cp_all_point[:, 0]-leaf[0])**2+(cp_all_point[:, 1]-leaf[1])**2+(cp_all_point[:, 2]-leaf[2])**2)
    #         cluster_idx = np.where(radius < max_radius)
    #         cluster.extend(idx[cluster_idx[0][:]])
    #         next_idx = np.delete(idx, cluster_idx[0])
    #         next_cp_all_point = np.delete(cp_all_point, cluster_idx[0], 0)
    #         for i in list(cluster_idx[0][:]):
    #             points.put(cp_all_point[i])
    #         idx = next_idx
    #         cp_all_point = next_cp_all_point   
    #     que.put(cluster)


if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequence = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    fuse = Fusion(root, ckpt)
    fuse.calibration()
