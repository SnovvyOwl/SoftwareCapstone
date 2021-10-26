import numpy as np
import pickle
from mmdet.apis import inference_detector ,init_detector
import os
import mmdet.core.visualization
from PIL import Image

labels_to_names_seq = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                       11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                       21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                       31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                       41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                       51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'sofa', 58: 'pottedplant', 59: 'bed', 60: 'diningtable',
                       61: 'toilet', 62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
                       71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

WAYMO_CLASSES = {  'Vehicle':0, 'Pedestrian':1,  'Sign':2, 'Cyclist':3}

WAYMO_CLASS = ['Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
class FastRCNN_sequence(object):
    def __init__(self,rootdir,seq,outputdir,saveimg=True,_gtcompare=True):
        self.sequence=seq
        self.root=rootdir
        self.img_list=os.listdir(self.root+self.sequence+"/img/")
        self.img_list.sort()
        self.outdir=outputdir+sequencename+'/'
        self.saveflag=saveimg
        self.gtcompare=_gtcompare
        self.GT_front=[]
        self.GT_front_left=[]
        self.GT_front_right=[]
        self.GT_side_left=[]
        self.GT_side_right=[]
        self.comparedir=self.outdir+"gt/"
        if not os.path.exists(self.comparedir):
            os.makedirs(self.comparedir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def load_gt(self):
        gtpath=self.root+self.sequence+"/img/"+self.img_list[0]
        with open( gtpath, 'rb') as f:
            groundtruth=pickle.load(f)
        for i in range(len(groundtruth)):
            if i%5==0:
                
                self.GT_front.append(groundtruth[i])
            elif i%5==1:
                self.GT_front_left.append(groundtruth[i])
            elif i%5==2:
                self.GT_front_right.append(groundtruth[i])
            elif i%5==3:
                self.GT_side_left.append(groundtruth[i])
            elif i%5==4:
                self.GT_side_right.append(groundtruth[i])
            

    def save(self,model):
        for file in self.img_list:
            if file[-1] == "g":
                img_name = self.root+self.sequence+"/img/"+file
                result=inference_detector(model, img_name)
                if  self.saveflag:
                    model.show_result(img_name, result, score_thr=0.3,show=False,wait_time=0,win_name=file,bbox_color=(72, 101, 241),text_color=(72, 101, 241),out_file=self.outdir+file[0:-4]+".jpg")
                np.save(self.outdir+file[0:-4]+".npy",result)
                if self.gtcompare:
                    self.compare_GT(img_name,result,file)
                    
                   
    def compare_GT(self,img_name,result,file):
        self.load_gt()
        i=file.find("S")
        if i!=-1:
            camera_num=file[i:-9]
        else:
            i=file.find("F")
            camera_num=file[i:-9]
        file_num=int(file[-8:-4])
        label=[]
        if camera_num=="FRONT_IMAGE":
            GT=self.GT_front[file_num]["ann"]
            GT["gt_bboxes"]=GT.pop("bboxes")
            for i,change in enumerate(GT["labels"]):
                label.append(int(WAYMO_CLASSES[change]))
            GT["gt_labels"]=np.array(label)

            # img=mmdet.core.visualization.imshow_gt_det_bboxes(img_name,GT,result,WAYMO_CLASS,0.3)
            # img=Image.fromarray(img)

        elif camera_num=="FRONT_LEFT_IMAGE":
            print()
        elif camera_num=="FRONT_RIGHT_IMAGE":
            print()
        elif camera_num=="SIDE_LEFT_IMAGE":
            print()
        else:
            print()
        # print(self.GT_front)
       


if __name__=="__main__":
    
    # Choose to use a config and initialize the detector
    config = './FasterRCNN/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = './FasterRCNN/checkpoints/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth'
    rootdir = "./PV_R-CNN/data/waymo/waymo_processed_data/"
    outputdir="./FasterRCNN/output/"
    model=init_detector(config, checkpoint, device='cuda:0')
    sequencename="segment-1024360143612057520_3580_000_3600_000_with_camera_labels"
    seq=FastRCNN_sequence(rootdir,sequencename,outputdir)
    seq.save(model)
    # seq.compare_GT()