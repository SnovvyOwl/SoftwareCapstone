import pickle
import copy
from re import M
import matplotlib
import numpy as np
from numpy.linalg.linalg import norm
import open3d as o3d
from tensorflow.python.eager.context import PhysicalDevice
from queue import Queue
import matplotlib.pyplot as plt
import torch
class viewbox(object):
    def __init__(self,minx,miny,maxx,maxy):
        self.min=np.array([[minx],[miny],[1]])
        self.max=np.array([[maxx],[maxy],[1]])
        self.boxmat=np.concatenate((self.min,self.max) ,axis=1)

def projection(intrinsic,extrinsic,lidar,weight,height):
    camera=viewbox(0,0,weight,height)
    physical_plane=np.matmul(np.linalg.inv(intrinsic),camera.boxmat).T
    one=np.ones((len(lidar),1))
    cp_lidar=np.concatenate((lidar,one),axis=1)
    cam=o3d.geometry.LineSet()
    campoint=[[0,0,0],[physical_plane[0,0]*50,physical_plane[0,1]*50,50],[physical_plane[0,0]*50,physical_plane[1,1]*50,50],[physical_plane[1,0]*50,physical_plane[0,1]*50,50],[physical_plane[1,0]*50,physical_plane[1,1]*50,50]]
    line=[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[2,4],[3,4]]
    cam.lines = o3d.utility.Vector2iVector(line)
    cam.points = o3d.utility.Vector3dVector(campoint)
    to_plane=np.matmul(np.linalg.inv(extrinsic)[0:3,:],cp_lidar.T).T
    # to_plane=np.matmul(extrinsic[0:3,:],cp_lidar.T).T
    to_plane=np.concatenate(([-1*to_plane[:,1].T],[-1*to_plane[:,2].T],[to_plane[:,0].T]),axis=0).T
    toplaneidx=[]
    for i,point in enumerate(to_plane):
        if point[2]>0:
            toplaneidx.append(i)
    normal=to_plane
    normal=normal/normal[:,2][:,None]
    in_box_point=[]
    idx=[]
    for i in toplaneidx:
        if normal[i,0]>physical_plane[0,0]:
            if normal[i,1]>physical_plane[0,1]:
                if normal[i,0]<physical_plane[1,0]:
                    if normal[i,1]<physical_plane[1,1]:
                        idx.append(i)
    return in_box_point,idx

def segmentation(frustrum,idx):
    cluster=[]
    # centroid_vector=[(frustrum[:,0].max()+frustrum[:,0].min())/2,(frustrum[:,1].max()+frustrum[:,1].min())/2,(frustrum[:,2].max()+frustrum[:,2].min())/2]
    radius=(frustrum[:,0]**2+frustrum[:,1]**2+frustrum[:,2]**2)**0.5
    radius=radius[np.argmin(radius)]
    mean_radius=(frustrum[:,0].mean()**2+frustrum[:,1].mean()**2+frustrum[:,2].mean()**2)**0.5
    centroid_vector=[frustrum[:,0].mean()/mean_radius*radius,frustrum[:,1].mean()/mean_radius*radius,(frustrum[:,2]).mean()/mean_radius*radius]
    vec=o3d.geometry.LineSet()
    vecpoint=[[0,0,0],centroid_vector]
    line=[[0,1]]
    vec.lines= o3d.utility.Vector2iVector(line)
    vec.points= o3d.utility.Vector3dVector(vecpoint)
    centroid,centorid_idx,fi=find_centroid(centroid_vector,frustrum,idx)
    cluster=make_cluster(centroid,fi,frustrum,idx,0.01)   
    return cluster,vec

def make_cluster(centroid,centroid_idx,frustrum,idx,max_radius=0.1):
    cluster=[]
    points=Queue()
    idices=Queue()
    points.put(centroid)
    idices.put(idx[centroid_idx])
    cluster.append(idx[centroid_idx])
    cp_frustrum=frustrum.copy()
    cp_idx=idx.copy()
    cp_frustrum=np.delete(cp_frustrum,centroid_idx,0)
    cp_idx=np.delete(cp_idx,centroid_idx)
    while points.empty()!=True:
        leaf=points.get()
        radius=np.array((cp_frustrum[:,0]-leaf[0])**2+(cp_frustrum[:,1]-leaf[1])**2+(cp_frustrum[:,2]-leaf[2])**2)
        idx=np.where(radius<max_radius)
        cluster.extend(cp_idx[idx[0][:]])
        next_cp_idx=np.delete(cp_idx,idx[0])
        next_cp_frustrum=np.delete(cp_frustrum,idx[0],0)
        for i in list(idx[0][:]):
            points.put(cp_frustrum[i])
        cp_idx=next_cp_idx
        cp_frustrum=next_cp_frustrum
    return cluster

def make_3dBox(anno):
    boxes=[]
    lines=[[0,1],[1,2],[2,3],[0,3],[0,4],[4,5],[5,6],[6,7],[4,7],[1,5],[2,6],[3,7]]
    corners3d=boxes_to_corners_3d(anno["boxes_lidar"])
    for box in corners3d:
        box3d=o3d.geometry.LineSet()
        box3d.points= o3d.utility.Vector3dVector(box)
        box3d.lines=o3d.utility.Vector2iVector(lines)
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

def find_centroid(center_vector,frustrum,idx):
    centroid=None
    radius=(frustrum[:,1]-center_vector[1])**2+(frustrum[:,2]-center_vector[2])**2
    centroid=frustrum[np.argmin(radius)]
    centroid_idx=idx[np.argmin(radius)]
    return centroid,centroid_idx ,np.argmin(radius)

if __name__ == "__main__":
    with open("anno3d.pkl",'rb')as f:
        annos3d=pickle.load(f)

    with open("anno2d.pkl",'rb')as f:
        annos2d=pickle.load(f)
    with open("frustrum.pkl",'rb')as f:
        frustrum=pickle.load(f)
    # generate some neat n times 3 matrix using a variant of sync function

    xyz=np.load("/home/seongwon/SoftwareCapstone/data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/0000.npy")
    xyz=xyz[:,:3]
  
    pcd = o3d.geometry.PointCloud()
    label =frustrum[0]["frustrum"][0]
    frustrum=xyz[label["frustrum"]]
    seg ,vec=segmentation(frustrum,label["frustrum"])

    pcd.points = o3d.utility.Vector3dVector( xyz[seg])
    box=make_3dBox(annos3d[0])
    # cameranum=4
    # # cameranum=int(input("input cameranum"))
    # if cameranum in [0,1,2]:
    #     point, idx=projection(annos2d[0]["intrinsic"][cameranum],annos2d[0]["extrinsic"][cameranum],xyz,1920,1280)
    # elif cameranum in [3,4]:
    #     point, idx =projection(annos2d[0]["intrinsic"][cameranum],annos2d[0]["extrinsic"][cameranum],xyz,1920,1280)
    cp_xyz= xyz.copy()
    cp_xyz=np.delete( xyz,seg,axis=0)
    # in_box_point= xyz[idx]
    # pcd = o3d.geometry.PointCloud()
    # 
    pcd.paint_uniform_color([1, 0.706, 0])
    # zero=o3d.geometry.PointCloud()
    # zero.points=o3d.utility.Vector3dVector(np.array([[0,0,0]]))
    # zero.paint_uniform_color([1,0,0])
    all=o3d.geometry.PointCloud()
    all.points=o3d.utility.Vector3dVector(cp_xyz)
    all.paint_uniform_color([0,0,1])
    point=[pcd,all]
    box.append(pcd)
    box.append(all)
    vis=o3d.visualization.Visualizer()
    vis.create_window()
    for b in box:
        vis.add_geometry(b)
    vis.add_geometry(all)
    vis.add_geometry(pcd)
    vis.get_render_option().line_width = 100
    vis.update_renderer()

    vis.run()
    vis.destroy_window()
