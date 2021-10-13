import numpy as np
import pickle
data=np.load("/home/seongwon/Softwarecaptsone/segment-17065833287841703_2980_000_3000_000_with_camera_labels/0003.npy") # point cloud;
new_cv=dict()
with open('/home/seongwon/Softwarecaptsone/segment-17065833287841703_2980_000_3000_000_with_camera_labels/segment-17065833287841703_2980_000_3000_000_with_camera_labels.pkl', 'rb') as f:  # cv.pkl이라는 파일을 바이너리 읽기(rb)모드로 열어서 f라 하고
    new_cv = pickle.load(f) 
print(data)
print(dict(zip(new_cv)))