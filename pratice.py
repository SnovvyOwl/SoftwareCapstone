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
    centroid,centorid_idx,fi=find_centroid(centroid_vector,frustrum,idx)
    points=[]
    idices=[]
    for i , point in enumerate(frustrum):
        if point[2]>0.1:
            points.append(point)
            idices.append(idx[i])
    cluster=make_cluster(centroid,fi,frustrum,idx,0.01)   
    # cluster=make_cluster(frustrum)   
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
        for i , point in enumerate(cp_frustrum):
            radius=np.array((point[0]-leaf[0])**2+(point[1]-leaf[1])**2+(point[2]-leaf[2])**2)
            if radius<max_radius:
                cluster.append(cp_idx[i])
                points.put(point)
                print(idx[i])
                next_cp_idx=np.delete(cp_idx,i)
                next_cp_frustrum=np.delete(cp_frustrum,i,0)
        cp_idx=next_cp_idx
        cp_frustrum=next_cp_frustrum
    return cluster
    
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