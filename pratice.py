import pickle
import copy
import numpy as np
from numpy.linalg.linalg import norm
import open3d as o3d
from tensorflow.python.eager.context import PhysicalDevice
class viewbox(object):
    def __init__(self,minx,miny,maxx,maxy):
        self.min=np.array([[minx],[miny],[1]])
        self.max=np.array([[maxx],[maxy],[1]])
        self.boxmat=np.concatenate((self.min,self.max) ,axis=1)

def projection(intrinsic,extrinsic,lidar,weight,height):
    turn1=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    turn2=np.array([[0,0,-1],[1,0,0],[0,-1,0]])
    camera=viewbox(0,0,weight,height)
    physical_plane=np.matmul(np.linalg.inv(intrinsic),camera.boxmat).T
    one=np.ones((len(lidar),1))
    cp_lidar=np.concatenate((lidar,one),axis=1)
    # cam=o3d.geometry.LineSet()
    # campoint=[[0,0,0],[physical_plane[0,0]*50,physical_plane[0,1]*50,50],[physical_plane[0,0]*50,physical_plane[1,1]*50,50],[physical_plane[1,0]*50,physical_plane[0,1]*50,50],[physical_plane[1,0]*50,physical_plane[1,1]*50,50]]
    # line=[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[2,4],[3,4]]
    # cam.lines = o3d.utility.Vector2iVector(line)
    # cam.points = o3d.utility.Vector3dVector(campoint)
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

if __name__ == "__main__":
    with open("anno3d.pkl",'rb')as f:
        annos3d=pickle.load(f)
    with open("anno2d.pkl",'rb')as f:
        annos2d=pickle.load(f)
    # generate some neat n times 3 matrix using a variant of sync function

    xyz=np.load("/home/seongwon/SoftwareCapstone/data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/0000.npy")
    xyz=xyz[:,:3]
    cameranum=4
    # cameranum=int(input("input cameranum"))
    if cameranum in [0,1,2]:
        point, idx=projection(annos2d[0]["intrinsic"][cameranum],annos2d[0]["extrinsic"][cameranum],xyz,1920,1280)
    elif cameranum in [3,4]:
        point, idx =projection(annos2d[0]["intrinsic"][cameranum],annos2d[0]["extrinsic"][cameranum],xyz,1920,1280)
    cp_xyz= xyz.copy()
    cp_xyz=np.delete( xyz,idx,axis=0)
    in_box_point= xyz[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(in_box_point)
    pcd.paint_uniform_color([1, 0.706, 0])
    # zero=o3d.geometry.PointCloud()
    # zero.points=o3d.utility.Vector3dVector(np.array([[0,0,0]]))
    # zero.paint_uniform_color([1,0,0])
    all=o3d.geometry.PointCloud()
    all.points=o3d.utility.Vector3dVector(cp_xyz)
    all.paint_uniform_color([0,0,1])
    o3d.visualization.draw_geometries([pcd,all])