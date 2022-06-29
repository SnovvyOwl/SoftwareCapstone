from re import I, T, match
import numpy as np
from numpy import ma
# from torch._C import float32
from modelmanager import ModelManager
import numpy as np
import torch
import pickle
from queue import Queue
from tqdm import tqdm
import PVRCNN.utils.common_utils
import PVRCNN.ops.roiaware_pool3d.roiaware_pool3d_utils
import math
from PVRCNN.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu ,boxes_inbox_gpu
import copy
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
    point_masks = PVRCNN.ops.roiaware_pool3d.roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points


def iou2d(box1, box2):
    '''
        DO : Calculate 2D IOU
        INPUT : box1[4], box2[4]  box = (x1, y1, x2, y2)
        OUTPUT : IoU   
    '''

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

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
        cosa, sina, zeros,
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
        # self.val = ModelManager(root, ckpt)
        self.current_intrinsics = None
        self.current_extrinsics = None
        self.root = root

    def make_extrinsic_mat(self, param):
        extrinscis = []
        for i in range(CAMMERA_NUM):
            extrinscis.append(param[i])
        return extrinscis

    def main(self,annos3d,annos2d):
        sequence = annos2d["frame_id"][0][0:-4]
        self.current_intrinsics = annos2d["intrinsic"]
        self.current_extrinsics = self.make_extrinsic_mat(annos2d["extrinsic"])
        
        fusion_res=copy.deepcopy(annos3d)
        img3d = {}
        if annos2d["frame_id"][0][0:-4] != sequence:
            self.current_intrinsics = annos2d["intrinsic"]
            self.current_extrinsics = self.make_extrinsic_mat(annos2d["extrinsic"])
            sequence=annos2d[0]["frame_id"][0][0:-4] 
        xyz = np.load(self.root + sequence + '/0' + annos2d["frame_id"][0][-3:] + ".npy")[:, :3]
        point_planes = self.pointcloud2image(xyz)
        frustum_for_onescene = self.make_frustum(annos2d["anno"], xyz, point_planes)
        box3d_to_2d = self.box_is_in_plane(annos3d[0])
        res = self.is_box_in_frustum(frustum_for_onescene, box3d_to_2d, xyz, annos3d[0]["boxes_lidar"])
        img3d["frustum"] = res
        img3d["segment_id"] = sequence
        img3d["frame_id"] = sequence + '_' + annos2d["frame_id"][0][-3:]
        img3d["filename"] = annos2d["image_id"]
    
        for label in res:
            if label["is_generated"] is True:
                if label["label"] in [ 'Vehicle', 'Pedestrian', 'Cyclist']:
                    fusion_res[0]["name"]=np.append(fusion_res[0]["name"],label["label"])
                    fusion_res[0]["score"]=np.append(fusion_res[0]["score"],np.array(min(fusion_res[0]["score"])))
                    fusion_res[0]["boxes_lidar"] = np.vstack((fusion_res[0]["boxes_lidar"],label["PVRCNN_Formed_Box"]))
        return img3d, annos3d ,fusion_res

    def pointcloud2image(self, lidar):
        '''
            DO: Point Cloud-> image Plane
            INPUT: Lidar PonitCloud
            OUTPUT: lidar point plane
        '''

        point_planes = []
        one = np.ones((len(lidar), 1))
        cp_lidar = np.concatenate((lidar, one), axis=1)
        cp_lidar = torch.from_numpy(cp_lidar).cuda()
        #  For Front Cameras
        for camera_num in range(3):
            to_plane = torch.matmul(
                torch.linalg.inv(torch.from_numpy(self.current_extrinsics[camera_num]).type(torch.float64).cuda()),
                cp_lidar.T).T
            change_coordinate = torch.stack([-1 * to_plane[:, 1], -1 * to_plane[:, 2], to_plane[:, 0]],
                                            dim=0).cpu().numpy().T
            lidar_idx = np.array((range(len(lidar))))
            negative_idx = np.where(change_coordinate[:, 2] <= 0)
            change_coordinate = np.delete(change_coordinate, negative_idx, axis=0).T
            lidar_idx = np.delete(lidar_idx, negative_idx)
            change_coordinate = torch.from_numpy(change_coordinate).type(torch.float64).cuda()
            to_image = torch.matmul(torch.from_numpy(self.current_intrinsics[camera_num]).type(torch.float64).cuda(),
                                    change_coordinate).T
            to_image = to_image / to_image[:, 2, None]
            point_plane = self.point_to_tensor(to_image, lidar_idx, 1920, 1280)  # Front Image Size(1920,1280)
            point_planes.append(point_plane.numpy())

        #  For Side Cameras
        for camera_num in [3, 4]:
            to_plane = torch.matmul(
                torch.linalg.inv(torch.from_numpy(self.current_extrinsics[camera_num]).type(torch.float64).cuda()),
                cp_lidar.T).T
            change_coordinate = torch.stack([-1 * to_plane[:, 1], -1 * to_plane[:, 2], to_plane[:, 0]],
                                            dim=0).cpu().numpy().T
            lidar_idx = np.array((range(len(lidar))))
            negative_idx = np.where(change_coordinate[:, 2] <= 0)
            change_coordinate = np.delete(change_coordinate, negative_idx, axis=0).T
            lidar_idx = np.delete(lidar_idx, negative_idx)
            change_coordinate = torch.from_numpy(change_coordinate).type(torch.float64).cuda()
            to_image = torch.matmul(torch.from_numpy(self.current_intrinsics[camera_num]).type(torch.float64).cuda(),
                                    change_coordinate).T
            to_image = to_image / to_image[:, 2, None]
            point_plane = self.point_to_tensor(to_image, lidar_idx, 1920, 886)  # Side Image Size (1920,886)
            point_planes.append(point_plane.numpy())
        return point_planes

    def point_to_tensor(self, calibrated_point, lidar_idx, width, height):
        '''
            DO: Calibrated point -> Image Plane 
            INPUT: calibrated_point = Traslation to image plane,
                    lidar_idx= lidar index in point cloud
                    width = box width
                    height =box height
            OUTPUT: point_image = Projected Point plane 
        '''
        point_image = -1 * torch.ones([width, height, 4], dtype=torch.int32, device='cuda:0')
        idx_tensor = torch.zeros([width, height], dtype=torch.int32)
        for idx, point in enumerate(calibrated_point):
            pixel_x = int(torch.floor(point[0]))
            pixel_y = int(torch.floor(point[1]))
            if pixel_x > 0 and pixel_x < width:
                if pixel_y > 0 and pixel_y < height:
                    point_image[pixel_x][pixel_y][idx_tensor[pixel_x][pixel_y]] = lidar_idx[idx]
                    idx_tensor[pixel_x][pixel_y] += 1
        return point_image.cpu()

    def make_frustum(self, annos, xyz, point_planes):
        ''' 
            DO: make Frustum , find Centroid of Frustum
            INPUT: annos=2d Object result
                    xyz=point cloud
                    point_planes= point cloud to image result
            OUTPUT: frustums =list of dict()
                        ["label"]= label
                        ["large_frustum"] =10 % large Frustum
                        ["frustum"]=frustum
                        ["centroid"]=centroid of Frustum
                        ["centroid_idx"]=centroid index
                        ["2d_box"]=2D object Dectection result
        '''
        frustums = []
        for camera_num in range(CAMMERA_NUM):
            for i, label in enumerate(annos[camera_num]["labels"]):
                idx = None
                if label != "unknown":
                    if annos[camera_num]['scores'][i] > 0.7:  # Erase not Critical Object
                        projected_point = {}
                        projected_point['score'] = annos[camera_num]['scores'][i]
                        projected_point["label"] = label
                        box = annos[camera_num]["boxes"][i]
                        box = np.floor(box).astype(np.int)  # Change Float to int
                        frustum = np.unique(point_planes[camera_num][box[0]:box[2], box[1]:box[3]].flatten("C"))
                        idx = np.where(frustum == -1)
                        if idx is not None:
                            frustum = np.delete(frustum, idx)
                        # make 10% Large Frustum
                        x_extend = ((box[2] - box[0]) * 1.1) / 2
                        x_center = (box[2] + box[0]) / 2
                        y_extend = ((box[3] - box[1]) * 1.1) / 2
                        y_center = (box[3] + box[1]) / 2
                        large_box = [x_center - x_extend, y_center - y_extend, x_center + x_extend, y_center + y_extend]
                        large_box = np.floor(large_box).astype(np.int)
                        large_frustum = np.unique(
                            point_planes[camera_num][large_box[0]:large_box[2], large_box[1]:large_box[3]].flatten("C"))
                        idx = np.where(large_frustum == -1)
                        if idx is not None:
                            large_frustum = np.delete(large_frustum, idx)

                        projected_point["large_frustum"] = large_frustum
                        projected_point["frustum"] = frustum
                        projected_point["2d_box"] = box
                        if frustum.size != 0:
                            # make center box
                            x_center_extend = ((box[2] - box[0]) * 0.1) / 2
                            y_center_extend = ((box[3] - box[1]) * 0.1) / 2
                            center_box = [x_center - x_center_extend, y_center - y_center_extend,
                                          x_center + x_center_extend, y_center + y_center_extend]
                            center_box = np.floor(center_box).astype(np.int)
                            center_frustum = np.unique(point_planes[camera_num][center_box[0]:center_box[2],
                                                       center_box[1]:center_box[3]].flatten("C"))
                            idx = np.where(center_frustum == -1)
                            if idx is not None:
                                center_frustum = np.delete(center_frustum, idx)
                            x_center_extend_L = ((box[2] - box[0]) * 0.2) / 2
                            y_center_extend_L = ((box[3] - box[1]) * 0.2) / 2
                            center_box_L = [x_center - x_center_extend_L, y_center - y_center_extend_L,
                                          x_center + x_center_extend_L, y_center + y_center_extend_L]
                            center_box_L = np.floor(center_box_L).astype(np.int)
                            center_frustum_L = np.unique(point_planes[camera_num][center_box_L[0]:center_box_L[2],
                                                       center_box_L[1]:center_box_L[3]].flatten("C"))
                            idx = np.where(center_frustum_L == -1)
                            if idx is not None:
                                center_frustum_L = np.delete(center_frustum_L, idx)
                            if len(center_frustum) != 0:
                                # divide part 
                                idxmin,idxmax=self.center_box_seg(xyz[center_frustum],center_frustum)
                                # If center box has 2 Object? erase Not Interest Object
                                if (len(idxmax)+len(idxmin))==len(center_frustum):
                                    if len(idxmin) != len(center_frustum):
                                        if len(idxmin) < len(idxmax):
                                            center_frustum = np.array(list(set(center_frustum) - set(idxmin)))
                                        elif len(idxmin) > len(idxmax):
                                            center_frustum = np.array(list(set(center_frustum) - set(idxmax)))
                                        else:
                                            idxmin,idxmax=self.center_box_seg( xyz[center_frustum_L],center_frustum_L)
                                            if len(idxmin) != len(center_frustum_L):
                                                if len(idxmin) < len(idxmax):
                                                    center_frustum = np.array(list(set(center_frustum_L) - set(idxmin)))
                                                elif len(idxmin) > len(idxmax):
                                                    center_frustum = np.array(list(set(center_frustum_L) - set(idxmax)))
                                                else:
                                                    center_frustum = np.array(list(set(idxmax)))
                                    else:
                                        center_frustum=np.array(list(set(idxmax)|(set(center_frustum_L) - set(idxmin)-set(idxmax))))

                                elif len(idxmax) != len(center_frustum):
                                    if (len(idxmax)+len(idxmin))==len(center_frustum):
                                        if len(idxmin) < len(idxmax):
                                            center_frustum = np.array(list(set(center_frustum) - set(idxmin)))
                                        elif len(idxmin) > len(idxmax):
                                            center_frustum = np.array(list(set(center_frustum) - set(idxmax)))
                                        else:
                                            idxmin,idxmax=self.center_box_seg( xyz[center_frustum_L],center_frustum_L)
                                            if len(idxmin) != len(center_frustum_L):
                                                if len(idxmin) < len(idxmax):
                                                    center_frustum = np.array(list(set(center_frustum_L) - set(idxmin)))
                                                elif len(idxmin) > len(idxmax):
                                                    center_frustum = np.array(list(set(center_frustum_L) - set(idxmax)))
                                                else:
                                                    center_frustum = np.array(list(set(idxmax)))
                                    else:
                                        center_frustum=np.array(list(set(idxmax)|( set(center_frustum_L) - set(idxmin)-set(idxmax))))

                                else: 

                                    pass
                            if len(xyz[center_frustum]) != 0:
                                projected_point["centroid"] = xyz[center_frustum]
                                projected_point["centroid_idx"] = center_frustum
                            else:  # Center Box is Empty
                                projected_point["centroid"] = None
                                projected_point["centroid_idx"] = None
                        else:  # frustum is Empty
                            projected_point["centroid"] = None
                            projected_point["centroid_idx"] = None

                        frustums.append(projected_point)
        return frustums

    def center_box_seg(self,center_box_point,center_box_idx):  
        radius = np.array((center_box_point[:, 0]) ** 2 + (center_box_point[:, 1]) ** 2 + (center_box_point[:, 2]) ** 2)
        min_radius_point = center_box_point[np.argmin(radius)]
        max_radius_point = center_box_point[np.argmax(radius)]
        min_radius_point = np.concatenate((min_radius_point, np.array([0, 0, 0])), axis=0)
        max_radius_point = np.concatenate((max_radius_point, np.array([0, 0, 0])), axis=0)
        min_radius_point = np.reshape(min_radius_point, (2, 3))
        max_radius_point = np.reshape(max_radius_point, (2, 3))
        resmin, idxmin = self.segmentation(center_box_point, min_radius_point,center_box_idx,seg_frustum=False,max_radius=0.05)
        resmax, idxmax = self.segmentation(center_box_point, max_radius_point,center_box_idx,seg_frustum=False,max_radius=0.05)
        return idxmin,idxmax

    def box_is_in_plane(self, annos):
        '''
            DO:  3D box ->2D Image
            INPUT: PV-RCNN 3D Boxes Oneframe
            OUTPUT: camera_num, pixel
        '''

        cp_box = annos["boxes_lidar"].copy()
        box_center = cp_box[:, :3].copy()
        box_plane = self.pointcloud2image(box_center)
        res = []
        for camera_num in range(CAMMERA_NUM):
            plane = []
            maked_2dbox_with_label = {}
            box_idx = np.unique(box_plane[camera_num].flatten("C"))
            idx = np.where(box_idx == -1)
            if idx is not None:
                box_idx = np.delete(box_idx, idx)
            corners_3d = boxes_to_corners_3d(cp_box[box_idx, :])
            for i, box in enumerate(corners_3d):
                box2d = self.make_2d_box(camera_num, box)
                maked_2dbox_with_label["3d_box"] = box
                maked_2dbox_with_label["box"] = box2d
                maked_2dbox_with_label["label"] = annos["name"][box_idx[i]]
                maked_2dbox_with_label["PVRCNN_Formed_Box"] = annos["boxes_lidar"][box_idx[i]]
                plane.append(maked_2dbox_with_label.copy())

            res.append(plane)
        return res

    def make_2d_box(self, camera_num, box):
        '''
            DO: 3D BOX -> 2D BOX for Each Camera Image
            INPUT: camera_num = number of Camera
            OUTPUT: box=3d box Result PV-RCNN
        '''
        cp_box = box.copy()
        one = np.ones((len(box), 1))
        cp_box = np.concatenate((cp_box, one), axis=1)
        to_plane = np.matmul(np.linalg.inv(self.current_extrinsics[camera_num]), cp_box.T).T
        change_coordinate = np.concatenate(([-1 * to_plane[:, 1].T], [-1 * to_plane[:, 2].T], [to_plane[:, 0].T]),
                                           axis=0)
        to_image = np.matmul(self.current_intrinsics[camera_num], change_coordinate).T
        to_image = to_image / to_image[:, 2, None]
        box = [np.min(to_image[:, 0]), np.min(to_image[:, 1]), np.max(to_image[:, 0]), np.max(to_image[:, 1])]
        return box

    def is_box_in_frustum(self, frustum_per_onescene, boxes, xyz, pvrcnn_box):
        '''
            DO: generated 2d Box(3D_box -> 2d box) is in frustum?
            INPUT: frustum_per_onescene = Frustum for One Frame
                    boxes= generated 2D Box
                    xyz=Point Cloud
            OUTPUT: frustum_per_onescene
                        ["label"]= label
                        ["large_frustum"] =10 % large Frustum
                        ["frustum"]=frustum
                        ["centroid"]=centroid of Frustum
                        ["centroid_idx"]=centroid index
                        ["2d_box"]=2D object Dectection Result
                        ["3d_box"]=3D Object Dectection Result or Genereate 3D Box
                                        7 -------- 4
                                       /|         /|
                                      6 -------- 5 .
                                      | |        | |
                                      . 3 -------- 0
                                      |/         |/
                                      2 -------- 1
                        ["is_generated"] = Box is Generated?
        '''
        cp_pvrcnn=pvrcnn_box
        cp_xyz = xyz.copy()
        for i, frustum in enumerate(frustum_per_onescene):
            found = False
            for box_in_camera_num in boxes:
                for box in box_in_camera_num:
                    if frustum["label"] == box["label"]:
                        iou = iou2d(box["box"], frustum["2d_box"])
                        if iou > 0.4:
                            # print(iou)
                            frustum_per_onescene[i]["PVRCNN_Formed_Box"] = box["PVRCNN_Formed_Box"]
                            frustum_per_onescene[i]["3d_box"] = box["3d_box"]
                            frustum_per_onescene[i]["is_generated"] = False
                            found = True
                            break
            if found is False:
                if frustum["centroid"] is not None:
                    gen_box,gen_seg,gen_PVRCNNbox,gen_PVRCNNbox1_3= self.make_3d_box(cp_xyz[frustum["large_frustum"]], frustum["centroid"],frustum["large_frustum"],frustum["label"])
                    if gen_box is not None:
                        frustum_per_onescene[i]["3d_box"]=gen_box
                        frustum_per_onescene[i]["seg"]=gen_seg
                        frustum_per_onescene[i]["PVRCNN_Formed_Box"]=gen_PVRCNNbox.astype(np.float32)
                        matched_box = self.is_box_in_box(frustum_per_onescene[i]["PVRCNN_Formed_Box"],cp_pvrcnn,frustum_per_onescene[i]["label"])
                        if matched_box is not None:
                            frustum_per_onescene[i]["is_generated"] = False
                            frustum_per_onescene[i]["PVRCNN_Formed_Box"] = matched_box
                            frustum_per_onescene[i]["3d_box"] = boxes_to_corners_3d(matched_box)
                        else:
                            frustum_per_onescene[i]["is_generated"] = True
                            frustum_per_onescene[i]["PVRCNN_Formed_Box"]=gen_PVRCNNbox1_3.astype(np.float32)
                            cp_pvrcnn=np.vstack((cp_pvrcnn,gen_PVRCNNbox.astype(np.float32)))
                    else:
                        frustum_per_onescene[i]["3d_box"]=gen_box
                        frustum_per_onescene[i]["is_generated"]=False

                else:
                    frustum_per_onescene[i]["is_generated"] = False
                    frustum_per_onescene[i]["3d_box"] = None
        return frustum_per_onescene

    @staticmethod
    def find_max(gt_idx, iou_mat):
        return gt_idx[np.argmax(iou_mat[0, gt_idx])]

    def is_box_in_box(self, generate_Box, PVRCNN_boxes,label):
        generate_Box = np.vstack((generate_Box, np.zeros(7)))
        mat = boxes_inbox_gpu(torch.tensor(generate_Box.astype("float32")).cuda(), torch.tensor(PVRCNN_boxes).cuda())
        # mat = boxes_iou3d_gpu(torch.tensor(generate_Box.astype("float32")).cuda(), torch.tensor(PVRCNN_boxes).cuda())
        mat = mat.cpu().numpy()
        match = np.where(mat[0] > 0.0)[0]
        
        if len(match) > 1:
            match = np.array([self.find_max(match, mat)])
        if len(match) != 0:
            if mat[0, int(match)] > 0.7:
                return PVRCNN_boxes[match]
            else:
          
                dx = PVRCNN_boxes[match][0, 3]
                dy = PVRCNN_boxes[match][0, 4]
                dz = PVRCNN_boxes[match][0, 5]

                box3d_gen = boxes_to_corners_3d(generate_Box)
                match_ceter_to_corner = ((box3d_gen[0][:, 0] - PVRCNN_boxes[match][0][0]) ** 2 + (box3d_gen[0][:, 1] - PVRCNN_boxes[match][0][1]) ** 2 + (box3d_gen[0][:, 2] - PVRCNN_boxes[match][0][2]) ** 2) ** 0.5
                box_radius = ((dx / 2) ** 2 + (dy / 2) ** 2 + (dz / 2) ** 2) ** 0.5
                
                if (len(np.where(match_ceter_to_corner < box_radius)[0])) >= 8:
                    return PVRCNN_boxes[match]
            
                else:
                    return None
        else:
            return None

    def make_3d_box(self, frustum_point, centroid_point, frustum_idx,name):
        """
            DO: Make 3D box from Segmentation Result
            INPUT: Frustum_point = 10% Large Frustum point [x,y,z] 
                    centroid_point=Point in Center box [x,y,z]
                    frustum_idx= frustum index from Point Cloud 
            OUTPUT:
         
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
        seg_cluster, seg_idx = self.segmentation(frustum_point, centroid_point, frustum_idx, max_radius=0.01)
        if seg_idx is None:
            return None,None,None,None
        elif len(seg_idx)<15:
            return None, None, None,None
        else:
            # *********************************************************************************************
            # PCA

            # Calculate Center Point

            center_x = (np.max(seg_cluster[:][:, 0]) + np.min(seg_cluster[:][:, 0])) / 2
            center_y = (np.max(seg_cluster[:][:, 1]) + np.min(seg_cluster[:][:, 1])) / 2
            center_z = (np.max(seg_cluster[:][:, 2]) + np.min(seg_cluster[:][:, 2])) / 2

            # Calculate  Covirence Matrix
            M20 = np.dot((seg_cluster[:, 0] - center_x).T, (seg_cluster[:, 0] - center_x))
            M11 = np.dot((seg_cluster[:, 0] - center_x).T, (seg_cluster[:, 1] - center_y))
            M02 = np.dot((seg_cluster[:, 1] - center_y).T, (seg_cluster[:, 1] - center_y))
            M = np.array([[M20, M11], [M11, M02]])
            # Diagonalization
            w, v = np.linalg.eig(M)

            # Check Principal Axis
            if w[0] > w[1]:
                axis = v[0]
            else:
                axis = v[1]
            # *********************************************************************************************
            # Calcluate Heading Angle
            heading = math.atan2(axis[1], axis[0])

            # Make Rotational Matrix
            cos_theta = math.cos(heading)
            sin_theta = math.sin(heading)
            mat_T = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])

            # Calculate Non Rotated Point
            rot_center = np.matmul(mat_T, np.array([center_x, center_y, center_z]).T)
            rot_points = np.matmul(mat_T, seg_cluster.T).T

            # Calculate Length of X,Y,Z
            dx = np.max(rot_points[:, 0] - rot_center[0]) - np.min(rot_points[:, 0] - rot_center[0])
            dy = np.max(rot_points[:, 1] - rot_center[1]) - np.min(rot_points[:, 1] - rot_center[1])
            dz = np.max(seg_cluster[:][:, 2]) - np.min(seg_cluster[:][:, 2])
            ratio=dy/dx

            if ratio<0.4:            ##Plane
                return None,None,None,None
            elif dy<0.25 or dx<0.25:  ## Column
                return None,None,None,None
            
            else:
                if name!="Sign":
                    if (center_z-dz/2)>1: ## is UPPER
                        return None,None,None,None
                # Result Form PV-RCNN
                res = np.array([center_x, center_y, center_z, dx, dy, dz,-heading])
                res2=np.array([center_x, center_y, center_z, dx*1.2, dy*1.2, dz*1.1,-heading])

            # Result Form Box
                to_box_mat = np.array(
                    [[cos_theta, -sin_theta, 0, center_x], [sin_theta, cos_theta, 0, center_y], [0, 0, 1, center_z]])
                template = np.array(
                    [[dx / 2, dy / 2, -dz / 2, 1], [dx / 2, -dy / 2, -dz / 2, 1], [-dx / 2, -dy / 2, -dz / 2, 1],
                    [-dx / 2, dy / 2, -dz / 2, 1], [dx / 2, dy / 2, dz / 2, 1], [dx / 2, -dy / 2, dz / 2, 1],
                    [-dx / 2, -dy / 2, dz / 2, 1], [-dx / 2, dy / 2, dz / 2, 1]])
                box = np.matmul(to_box_mat, template.T).T
                return box, seg_cluster, res, res2

    def segmentation(self, frustum_point, centroid_point, frustum_idx,seg_frustum=True, max_radius=0.01):
        '''
            DO: Segmentation (make Cluster)
            INPUT: frustum_point= 10% Large Frustum (N,3)[x,y,z]
                    centroid_point =Point in Center Box (n,3)[x,y,z]
                    frustum_idx =Frustum Index, Index From Point Cloud (N)
            OUTPUT: cluster=Segmentation Result (M,3)[x,y,z]  
                    idx= Segmentation Point Index (M) 
        '''
        # INIT
        points = Queue()
        cp_frustum_point = frustum_point.copy()
        cp_frustum_idx = frustum_idx.copy()
        cluster = []
        for center_point in centroid_point:
            points.put(center_point)
            cluster.append(center_point)
        idx = []
        while points.empty() != True:
            leaf = points.get()
            radius = np.array((cp_frustum_point[:, 0] - leaf[0]) ** 2 + (cp_frustum_point[:, 1] - leaf[1]) ** 2 + (
                        cp_frustum_point[:, 2] - leaf[2]) ** 2)
            cluster_idx = np.where(radius < max_radius)
            idx.extend(cp_frustum_idx[cluster_idx])
            next_cp_frustum_idx = np.delete(cp_frustum_idx, cluster_idx[0])
            next_cp_frustum_point = np.delete(cp_frustum_point, cluster_idx[0], 0)
            for i in list(cluster_idx[0][:]):
                cluster.append(cp_frustum_point[i])
                points.put(cp_frustum_point[i])
            cp_frustum_idx = next_cp_frustum_idx
            cp_frustum_point = next_cp_frustum_point
        cluster = np.array(cluster)
        if seg_frustum:
            if len(idx)!=len(cluster):
                return cluster, idx
            else:
                return None,None
        else:
            return cluster, idx

    # #Test Code
    # def set_matrix(self):
    #     with open("anno2d.pkl", 'rb')as f:
    #         annos2d = pickle.load(f)
    #     self.current_intrinsics = annos2d[0]["intrinsic"]
    #     self.current_extrinsics = self.make_extrinsic_mat(annos2d[0]["extrinsic"])

    # def doitwell(self,planes):
    #     frustums=[]
    #     for plane in planes:
    #         frustum=np.unique(plane.flatten("C"))
    #         idx = np.where(frustum == -1)
    #         if idx is not None:
    #             frustum = np.delete(frustum, idx)
    #         frustums.append(frustum)
    #     return frustums


if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequence = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    fuse = Fusion(root, ckpt)
    fuse.main()