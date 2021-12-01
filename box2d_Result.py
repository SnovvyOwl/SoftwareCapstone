from WaymoDataset import Waymo2DLoader
import pickle
import numpy as np
import cv2
import torch
if __name__=="__main__":
    root = "./data/waymo/waymo_processed_data/"
    sequece = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    wd=Waymo2DLoader(root,segment=sequece)
    img,target=wd.__getitem__(5)
    idx=4
    t_img=img[idx]
    print(target[idx])
    for i,box in enumerate(target[idx]['boxes']):
        box=box.numpy()
        if target[idx]["labels"][i]==1:
            cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0))
        if target[idx]["labels"][i]==2:
            cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0))
        if target[idx]["labels"][i]==3:
            cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255))
        if target[idx]["labels"][i]==4:
            cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,255))
        if target[idx]["labels"][i]==0:
            cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,255))
    cv2.imwrite('005side_right.jpg',t_img)
    cv2.waitKey()
    cv2.destroyAllWindows()