import pickle
import random
import copy
from re import M
import numpy as np
from numpy.linalg.linalg import norm
import open3d as o3d
from tensorflow.python.eager.context import PhysicalDevice
from queue import Queue
import torch
from fusion import Fusion
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
    # np.savetxt("pase",np.array(idx),fmt='%6.6e')
    return in_box_point,idx

# def segmentation(frustum,idx):
#     cluster=[]
#     # centroid_vector=[(frustum[:,0].max()+frustum[:,0].min())/2,(frustum[:,1].max()+frustum[:,1].min())/2,(frustum[:,2].max()+frustum[:,2].min())/2]
#     radius=(frustum[:,0]**2+frustum[:,1]**2+frustum[:,2]**2)**0.5
#     radius=radius[np.argmin(radius)]
#     mean_radius=(frustum[:,0].mean()**2+frustum[:,1].mean()**2+frustum[:,2].mean()**2)**0.5
#     centroid_vector=[frustum[:,0].mean()/mean_radius*radius,frustum[:,1].mean()/mean_radius*radius,(frustum[:,2]).mean()/mean_radius*radius]
#     vec=o3d.geometry.LineSet()
#     vecpoint=[[0,0,0],centroid_vector]
#     line=[[0,1]]
#     vec.lines= o3d.utility.Vector2iVector(line)
#     vec.points= o3d.utility.Vector3dVector(vecpoint)
#     centroid,centorid_idx,fi=find_centroid(centroid_vector,frustum,idx)
#     cluster=make_cluster(centroid,fi,frustum,idx,0.01)   
#     return cluster,vec

# def make_cluster(centroid,centroid_idx,frustum,idx,max_radius=0.1):
#     cluster=[]
#     points=Queue()
#     idices=Queue()
#     points.put(centroid)
#     idices.put(idx[centroid_idx])
#     cluster.append(idx[centroid_idx])
#     cp_frustum=frustum.copy()
#     cp_idx=idx.copy()
#     cp_frustum=np.delete(cp_frustum,centroid_idx,0)
#     cp_idx=np.delete(cp_idx,centroid_idx)
#     while points.empty()!=True:
#         leaf=points.get()
#         radius=np.array((cp_frustum[:,0]-leaf[0])**2+(cp_frustum[:,1]-leaf[1])**2+(cp_frustum[:,2]-leaf[2])**2)
#         i=np.where(radius<max_radius)
#         cluster.extend(cp_idx[i[0][:]])
#         next_cp_idx=np.delete(cp_idx,i[0])
#         next_cp_frustum=np.delete(cp_frustum,i[0],0)
#         for i in list(idx[0][:]):
#             points.put(cp_frustum[i])
#         cp_idx=next_cp_idx
#         cp_frustum=next_cp_frustum
#     return cluster

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

def find_centroid(center_vector,frustum,idx):
    centroid=None
    radius=(frustum[:,1]-center_vector[1])**2+(frustum[:,2]-center_vector[2])**2
    centroid=frustum[np.argmin(radius)]
    centroid_idx=idx[np.argmin(radius)]
    return centroid,centroid_idx ,np.argmin(radius)

def image2point(xyz,pointidx):
    res=[]
    vec=[]
    for i,f in enumerate(pointidx):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[f])
        pcd.paint_uniform_color([i/5, 0.1, 0])
        vec=vec+list(f)
        res.append(pcd)
    return res,vec

if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequence = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    fu=Fusion( root,ckpt)
    with open("anno3d.pkl",'rb')as f:
        annos3d=pickle.load(f)
    with open("anno2d.pkl",'rb')as f:
        annos2d=pickle.load(f)
    with open("frustum.pkl",'rb')as f:
        frustum=pickle.load(f)
    xyz=np.load("/home/seongwon/SoftwareCapstone/data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/0005.npy")
    xyz=xyz[:,:3]
    fu.set_matrix()
    fu.box_is_in_plane(annos3d[0])
    plane=fu.pointcloud2image(xyz)
    r=fu.make_frustum(annos2d[0]["anno"],xyz,plane)
    # frustum,id=image2point(xyz,res)
    segs=[]
    frustums=[]
    print(annos3d[1]["frame_id"])
    # for f in frustum[1]["frustum"]:
    #     if f["is_generated"] is True:
    #         segs.append(f["seg"])
    for f in frustum[1]["frustum"]:
        if f["centroid_idx"] is not None:
            frustums.append(f)
    pcds=[]
    vecs=[]
    # for seg in segs:
    #     pcd=o3d.geometry.PointCloud()
    #     pcd.points=o3d.utility.Vector3dVector(xyz[seg])
    #     pcd.paint_uniform_color([np.random.rand() , np.random.rand() , 0])   
    #     pcds.append(pcd)
    #     vecs=vecs+list(seg)
    for f in frustums:
        if f["label"]=='Pedestrian':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz[f["frustum"]])
            pcd.paint_uniform_color([1, 0 , 0])
            pcds.append(pcd)
            vecs=vecs+list(f["frustum"])
        elif f["label"]=='Vehicle':
            pcd= o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(xyz[f["frustum"]])
            pcd.paint_uniform_color([0,1, 0])
        elif f["label"]=='Cyclist':
            pcd= o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(xyz[f["frustum"]])
            pcd.paint_uniform_color([0, np.random.rand() , np.random.rand() ])
        # pcds.append(pcd)
        # vecs=vecs+list(f["frustum"])
    
    box=make_3dBox(annos3d[1])
    # cameranum=0
    # if cameranum in [0,1,2]:
    #     point, idx=projection(annos2d[0]["intrinsic"][cameranum],annos2d[0]["extrinsic"][cameranum],xyz,1920,1280)
    # elif cameranum in [3,4]:
    #     point, idx =projection(annos2d[0]["intrinsic"][cameranum],annos2d[0]["extrinsic"][cameranum],xyz,1920,1280)
    cp_xyz= xyz.copy()
    cp_xyz=np.delete(xyz,vecs,axis=0)
  
    all=o3d.geometry.PointCloud()
    all.points=o3d.utility.Vector3dVector(cp_xyz)
    all.paint_uniform_color([0,0,1])
    # point=o3d.geometry.PointCloud()
    # point.points=o3d.utility.Vector3dVector(xyz[idx])
    # point.paint_uniform_color([0,1,0])
    # o3d.visualization.draw_geometries([all,point])
    # res.append(all)
    vis=o3d.visualization.Visualizer()
    vis.create_window()
    for b in box:
        vis.add_geometry(b)
    vis.add_geometry(all)
    for f in pcds:
        vis.add_geometry(f)
    vis.get_render_option().line_width = 100
    vis.update_renderer()

    vis.run()
    vis.destroy_window()
