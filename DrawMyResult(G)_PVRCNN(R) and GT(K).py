import pickle
import numpy as np
import open3d as o3d
import torch
from fusion import Fusion

def make_3dBox(anno):
    boxes=[]
    lines=[[0,1],[1,2],[2,3],[0,3],[0,4],[4,5],[5,6],[6,7],[4,7],[1,5],[2,6],[3,7]]
    corners3d=boxes_to_corners_3d(anno["boxes_lidar"])
    for box in corners3d:
        box3d=o3d.geometry.LineSet()
        box3d.points= o3d.utility.Vector3dVector(box)
        box3d.lines=o3d.utility.Vector2iVector(lines)
        box3d.colors=o3d.utility.Vector3dVector(PVRCNNcolor)
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
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


if __name__ == "__main__":
    frame_idx=1
    dirpath="./data/waymo/waymo_infos_val.pkl"
    lines=[[0,1],[1,2],[2,3],[0,3],[0,4],[4,5],[5,6],[6,7],[4,7],[1,5],[2,6],[3,7]]
    fusion_color=[[0, 1, 0] for i in range(len(lines))]
    PVRCNNcolor=[[1, 0, 0] for i in range(len(lines))]
    ## GT BOX######################
    with open(dirpath,"rb") as f:
        gt_data=pickle.load(f)
    print(gt_data[5*frame_idx]["frame_id"])
    gt_boxes=gt_data[5*frame_idx]["annos"]["gt_boxes_lidar"]
    groudTruth_boxes=[]
    corners3d=boxes_to_corners_3d(gt_boxes)
    for box in corners3d:
        box3d=o3d.geometry.LineSet()
        box3d.points= o3d.utility.Vector3dVector(box)
        box3d.lines=o3d.utility.Vector2iVector(lines)
        groudTruth_boxes.append(box3d)
    ############################################

    
    with open("anno3d.pkl",'rb')as f:
        annos3d=pickle.load(f)
    with open("frustum.pkl",'rb')as f:
        frustum=pickle.load(f)
    xyz=np.load("/home/seongwon/SoftwareCapstone/data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/0005.npy")
    xyz=xyz[:,:3]
  
    segs=[]
    frustums=[]
    print(annos3d[frame_idx]["frame_id"])
    for f in frustum[1]["frustum"]:
        if f["is_generated"] is True:
            segs.append(f)
    pcds=[]
    vecs=[]
    boxes=[]
  
    for seg in segs:
        if seg["label"]=='Pedestrian':
            box3d=o3d.geometry.LineSet()
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(seg["seg"])
            pcd.paint_uniform_color([np.random.rand() , np.random.rand() , 0])
            box3d.points= o3d.utility.Vector3dVector(seg["3d_box"])
            box3d.lines=o3d.utility.Vector2iVector(lines)
            box3d.colors=o3d.utility.Vector3dVector(fusion_color)
            boxes.append(box3d)
            pcds.append(pcd)
            vecs=vecs+list(seg["seg"])
        if seg["label"]=='Vehicle':
    
            box3d=o3d.geometry.LineSet()
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(seg["seg"])
            pcd.paint_uniform_color([np.random.rand() , np.random.rand() , 0])
            #box=fu.make_3d_box(seg["seg"])
            box3d.points= o3d.utility.Vector3dVector(seg["3d_box"])
            box3d.lines=o3d.utility.Vector2iVector(lines)
            box3d.colors=o3d.utility.Vector3dVector(fusion_color)
            boxes.append(box3d)
            pcds.append(pcd)
            vecs=vecs+list(seg["seg"])
        if seg["label"]=='Cyclist':
            box3d=o3d.geometry.LineSet()
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(seg["seg"])
            pcd.paint_uniform_color([np.random.rand() , np.random.rand() , 0])
            #box=fu.make_3d_box(seg["seg"])
            box3d.points= o3d.utility.Vector3dVector(seg["3d_box"])
            box3d.lines=o3d.utility.Vector2iVector(lines)
            box3d.colors=o3d.utility.Vector3dVector(fusion_color)
            boxes.append(box3d)
            pcds.append(pcd)
            vecs=vecs+list(seg["seg"])
        elif seg["label"]=='Sign':
            box3d=o3d.geometry.LineSet()
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(seg["seg"])
            pcd.paint_uniform_color([np.random.rand() , np.random.rand() , 0])
            #box=fu.make_3d_box(seg["seg"])
            box3d.points= o3d.utility.Vector3dVector(seg["3d_box"])
            box3d.lines=o3d.utility.Vector2iVector(lines)
            box3d.colors=o3d.utility.Vector3dVector(fusion_color)
            boxes.append(box3d)
            pcds.append(pcd)
            vecs=vecs+list(seg["seg"])  
    box=make_3dBox(annos3d[frame_idx])
    cp_xyz= xyz.copy()

  
    all=o3d.geometry.PointCloud()
    all.points=o3d.utility.Vector3dVector(cp_xyz)
    all.paint_uniform_color([0,0,1])
    vis=o3d.visualization.Visualizer()
    vis.create_window()
    for b in groudTruth_boxes:
        vis.add_geometry(b)
    for b in box:
        vis.add_geometry(b)
    for b in boxes:
        vis.add_geometry(b)
    vis.add_geometry(all)
    for f in pcds:
        vis.add_geometry(f)
    vis.get_render_option().line_width = 100
    vis.update_renderer()

    vis.run()
    vis.destroy_window()
