import pickle
import copy
import numpy as np
import open3d as o3d
class viewbox(object):
    def __init__(self,minx,miny,maxx,maxy):
        self.min=np.array([[minx],[miny],[1]])
        self.max=np.array([[maxx],[maxy],[1]])
        self.boxmat=np.concatenate((self.min,self.max) ,axis=1)

def calibration(intrinsic,extrinsic,img,lidar):
    camera=viewbox(0,0,1920,1280)
    physical_plane=np.matmul(np.linalg.inv(intrinsic),camera.boxmat).T

    one=np.ones((len(lidar),1))
    cp_lidar=np.concatenate((lidar,one),axis=1)
    to_plane=np.matmul(extrinsic[0:3,:],cp_lidar.T)
    normal=to_plane.T
    normal=normal/normal[:,2][:,None]
    in_box_point=[]
    idx=[]
    for i in range(len(normal)):
        if normal[i,0]>physical_plane[0,0]:
            if normal[i,1]>physical_plane[0,1]:
                if normal[i,0]<physical_plane[1,0]:
                    if normal[i,1]<physical_plane[1,1]:
                        in_box_point.append(lidar[i])
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
    cameranum=1
    if cameranum==1:
        point,idx=calibration(annos2d[0]["intrinsic"][0],annos2d[0]["extrinsic"][0],annos2d[0]["imgs"][0],xyz)
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
   
    # cp_xyz=xyz.copy()
    # cp_xyz=np.delete(cp_xyz,idx,axis=0)
    # print(cp_xyz)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cp_xyz)
    # o3d.io.write_point_cloud("sync.ply", pcd)

    # # # # Load saved point cloud and visualize it
    # pcd_load = o3d.io.read_point_cloud("sync.ply")
    # o3d.visualization.draw_geometries([pcd_load])

    # # convert Open3D.o3d.geometry.PointCloud to numpy array
    # xyz_load = np.asarray(pcd_load.points)
    # print('xyz_load')
    # print(xyz_load)