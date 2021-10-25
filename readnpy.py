import numpy as np
import pickle
data=np.load("/home/seongwon/SoftwareCapstone/data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/img/extrinsic.npy") # point cloud;
new_cv=dict()
with open('/home/seongwon/SoftwareCapstone/data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/img/camera_segment-1024360143612057520_3580_000_3600_000_with_camera_labels.pkl', 'rb') as f:  # cv.pkl이라는 파일을 바이너리 읽기(rb)모드로 열어서 f라 하고
    new_cv = pickle.load(f) 
print(data)
print(new_cv)
# for i in range(199):