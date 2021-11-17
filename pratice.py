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

def segmentation(frustrum,label,box,idx):
    cluster=[]
    centroid_vector=[(frustrum[:,0].max()+frustrum[:,0].min())/2,(frustrum[:,1].max()+frustrum[:,1].min())/2,(frustrum[:,2].max()+frustrum[:,2].min())/2]
    centroid_vector=[frustrum[:,0].min(),frustrum[:,1].mean()/frustrum[:,0].mean()*frustrum[:,0].min(),(frustrum[:,2]).mean()/frustrum[:,0].mean()*frustrum[:,0].min()]
    vec=o3d.geometry.LineSet()
    vecpoint=[[0,0,0],centroid_vector]
    line=[[0,1]]
    vec.lines= o3d.utility.Vector2iVector(line)
    vec.points= o3d.utility.Vector3dVector(vecpoint)
    centroid,centorid_idx=find_centroid(centroid_vector,frustrum,idx)
    points=[]
    idices=[]
    for i , point in enumerate(frustrum):
        if point[2]>0.1:
            points.append(point)
            idices.append(idx[i])
    cluster=make_cluster(centroid,frustrum,idx,0.01)   
    # cluster=make_cluster(frustrum)   
    return cluster,vec

# def make_cluster(frustrum):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(frustrum)
#     plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                          ransac_n=3,
#                                          num_iterations=1000)
#     [a, b, c, d] = plane_model
#     print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

#     inlier_cloud = pcd.select_by_index(inliers)
#     inlier_cloud.paint_uniform_color([1.0, 0, 0])
#     outlier_cloud = pcd.select_by_index(inliers, invert=True)
#     o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=0.8,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])
def make_cluster(centroid,frustrum,idx,max_radius=2):
    cluster=[]
    points=Queue()
    idices=Queue()
    for i , point in enumerate(frustrum):
        radius=np.array((point[0]-centroid[0])**2+(point[1]-centroid[1])**2+(point[2]-centroid[2])**2)
        if radius<max_radius:
            cluster.append(idx[i])
        else:
            points.put(point)
            idices.put(idx[i])
    return cluster
    
def find_centroid(center_vector,frustrum,idx):
    centroid=None
    radius=(frustrum[:,1]-center_vector[1])**2+(frustrum[:,2]-center_vector[2])**2
    centroid=frustrum[np.argmin(radius)]
    centroid_idx=idx[np.argmin(radius)]
    return centroid,centroid_idx

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
    seg ,vec=segmentation(frustrum,label,annos2d[0]["anno"][0]["boxes"][0],label["frustrum"])
  
    pcd.points = o3d.utility.Vector3dVector(xyz[seg])

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
    o3d.visualization.draw_geometries([pcd,all,vec])