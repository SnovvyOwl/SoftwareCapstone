from WaymoDataset import Waymo2DLoader
import pickle
import numpy as np
import cv2
import torch

if __name__ == "__main__":
    frame_num = 5
    idx = 0

    root = "./data/waymo/waymo_processed_data/"
    sequece = 'segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    wd = Waymo2DLoader(root, segment=sequece)

    ### DATALOAD
    with open(
            "data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/img/camera_segment-1024360143612057520_3580_000_3600_000_with_camera_labels.pkl",
            "rb") as f:
        gt_data = pickle.load(f)
    with open("frustum.pkl", 'rb') as f:
        result_of_fusion = pickle.load(f)

    img, target = wd.__getitem__(frame_num)
    t_img = img[idx]
    gt = gt_data[frame_num * 5 + idx]['ann']
    # t_img=np.zeros(t_img.shape)
    gt_padd = 100

    maked_padd = 80
    # GT
    for i, label in enumerate(gt["labels"]):
        box = gt["bboxes"][i]
        if label == "Vehicle":
            cv2.rectangle(t_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (255 - gt_padd, gt_padd, gt_padd))
        if label == "Pedestrian":
            cv2.rectangle(t_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (gt_padd, 255, gt_padd))
        if label == "Sign":
            cv2.rectangle(t_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (gt_padd, gt_padd, 255 - gt_padd))
        if label == "Cyclist":
            cv2.rectangle(t_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (gt_padd, 255 - gt_padd, 255 - gt_padd))
        if label == "Unknown":
            cv2.rectangle(t_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (255 - gt_padd, gt_padd, 255 - gt_padd))
    # PREDICTION
    # for i,box in enumerate(target[idx]['boxes']):
    #     box=box.numpy()
    #     if target[idx]["labels"][i]==1:
    #         cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0))
    #     if target[idx]["labels"][i]==2:
    #         cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0))
    #     if target[idx]["labels"][i]==3:
    #         cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255))
    #     if target[idx]["labels"][i]==4:
    #         cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,255))
    #     if target[idx]["labels"][i]==0:
    #         cv2.rectangle(t_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,255))
    cv2.imwrite('005front.jpg', t_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
