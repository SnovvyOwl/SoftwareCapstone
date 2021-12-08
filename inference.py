import enum
import pickle
import numpy as np
import torch
from fusion import Fusion
from PVRCNN.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

class AveragePrecision(object):
    def __init__(self,_name):
        self.true_positive=np.array([])
        self.gt_len=0
        self.detection_len=0
        self.name=_name
        self.ap=0
    
    def frame_add(self,gt_len,detection_len):
        self.gt_len+=gt_len
        self.detection_len+=detection_len
    
    def add(self,score,tp,iou,match):
        if self.true_positive.shape[0]!=0:
            self.true_positive=np.vstack((self.true_positive,np.array([score,tp,iou,match])))
        else:
            self.true_positive=np.array([score,tp,iou,match])
       
    def get_AP(self):
        coffindence=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
        pre_recall=0
        for cs in coffindence:
            current=np.where(self.true_positive[:,0]>cs)[0]
            recall=len(np.where(self.true_positive[current,1]==1.0)[0])/self.gt_len
            precision=len(np.where(self.true_positive[current,1]==1.0)[0])/len(current)
            self.ap+=(recall-pre_recall)*precision
            pre_recall=recall
        return self.ap
    
class Inference(object):
    def __init__(self,root,ckptdir):
        self.root=root
        self.ckptdir=ckptdir
        self.current_segment=None
        self.current_segment_frame=None
        self.gt_data=None
        self.PVRCNN_result=None
        self.result_of_fusion=None
        self.updated_result=[]
        self.pedestrianAP=AveragePrecision("Pedestrian")
        self.vehicleAP=AveragePrecision("Vehicle")
        self.cyclistAP=AveragePrecision("Cyclist")
        # self.fusion=Fusion(root,ckptdir)
    
    def load_ground_truth_data(self):
        dirpath="./data/waymo/waymo_infos_val.pkl"
        with open(dirpath,"rb") as f:
            self.gt_data=pickle.load(f)
    @staticmethod
    def find_max(gt_idx,fram_idx,iou_mat):
        return gt_idx[np.argmax(iou_mat[fram_idx,gt_idx])]

    def main(self):
        self.load_ground_truth_data()
        # self.result_of_fusion,self.PVRCNN_result= self.fusion.main()
        # self.PVRCNN_result=annos3d
        with open("frustum.pkl",'rb')as f:
            self.result_of_fusion=pickle.load(f)
        with open("anno3d.pkl", 'rb')as f:
            self.PVRCNN_result = pickle.load(f)
        self.add_result()
        for frame in self.updated_result:
            for gt_frame in self.gt_data:
                if frame["frame_id"]==gt_frame["frame_id"]:
                    # TP
                    # Calcluate IoU For One Frame
                    sign_idx=np.where(gt_frame["annos"]["name"]=="Sign")
                    iou_mat=boxes_iou3d_gpu(torch.tensor(gt_frame["annos"]["gt_boxes_lidar"].astype("float32")).cuda(),torch.tensor(frame["boxes_lidar"]).cuda())
                    iou_mat=iou_mat.cpu().numpy()
                    iou_mat=np.delete(iou_mat,sign_idx,axis=0)
                    gt_name=np.delete(gt_frame["annos"]["name"],sign_idx,axis=0)
                    self.match_correction(gt_name,frame["name"],iou_mat,frame["score"])
                
                    break
                
        print(self.pedestrianAP.get_AP())
        print(self.vehicleAP.get_AP())
        print(self.cyclistAP.get_AP())
        # print("True Prediction: {0}".format(all_TP))
        # print("ALL_GT: {0}".format(all_gt))
        # print("ALL_Det: {0}".format(all_detection))
        # print("Precision : {0}".format(float(all_TP)/float(all_detection)))
        # print("Recall : {0}".format(float(all_TP)/float(all_gt)))
  

    def match_correction(self,gt_name,frame_name,_iou_mat,score,unmatched_threshold=0.5,matched_threshold=0.5):
        iou_mat=_iou_mat.T
        for i, name in enumerate(frame_name):
            if name =="Pedestrian":
                gt_idx=np.where(iou_mat[i]>0)[0]
                if len(gt_idx)==0:
                    self.pedestrianAP.add(score[i],False,0,False)
                else:
                    if len(gt_idx)>1:
                        gt_idx=self.find_max(gt_idx,i,iou_mat)
                    else:
                        gt_idx=gt_idx[0]
                    if gt_name[gt_idx]==frame_name[i]:
                        found=bool(iou_mat[i,gt_idx]>unmatched_threshold)
                        match=bool(iou_mat[i,gt_idx]>matched_threshold)
                    else:
                        found=False
                        match=False
                    self.pedestrianAP.add(score[i],found,iou_mat[i,gt_idx],match)
            elif name =="Vehicle":
                gt_idx=np.where(iou_mat[i]>0)[0]
                if len(gt_idx)==0:
                    self.vehicleAP.add(score[i],False,0,False)
                else:
                    if len(gt_idx)>1:
                        gt_idx=self.find_max(gt_idx,i,iou_mat)
                    else:
                        gt_idx=gt_idx[0]
                    if gt_name[gt_idx]==frame_name[i]:
                        found=bool(iou_mat[i,gt_idx]>unmatched_threshold)
                        match=bool(iou_mat[i,gt_idx]>matched_threshold)
                    else:
                        found=False
                        match=False
                    self.vehicleAP.add(score[i],found,iou_mat[i,gt_idx],match)
            elif name =="Cyclist":
                gt_idx=np.where(iou_mat[i]>0)[0]
                if len(gt_idx)==0:
                    self.cyclistAP.add(score[i],False,0,False)
                else:
                    if len(gt_idx)>1:
                        gt_idx=self.find_max(gt_idx,i,iou_mat)
                    else:
                        gt_idx=gt_idx[0]
                    if gt_name[gt_idx]==frame_name[i]:
                        found=bool(iou_mat[i,gt_idx]>unmatched_threshold)
                        match=bool(iou_mat[i,gt_idx]>matched_threshold)
                    else:
                        found=False
                        match=False
                    self.cyclistAP.add(score[i],found,iou_mat[i,gt_idx],match)  
        self.pedestrianAP.frame_add(len(np.where(gt_name=="Pedestrian")[0]),len(np.where(frame_name=="Pedestrian")[0]))
        self.vehicleAP.frame_add(len(np.where(gt_name=="Vehicle")[0]),len(np.where(frame_name=="Vehicle")[0]))
        self.cyclistAP.frame_add(len(np.where(gt_name=="Cyclist")[0]),len(np.where(frame_name=="Cyclist")[0]))

    def add_result(self): 
        for result_frame in self.result_of_fusion:
            update_frame={}
            for PV_RCNN_frame in self.PVRCNN_result:
                if PV_RCNN_frame["frame_id"]==result_frame["frame_id"]:
                    update_frame["name"]=PV_RCNN_frame["name"]
                    update_frame["score"]=PV_RCNN_frame["score"]
                    update_frame["frame_id"]=PV_RCNN_frame["frame_id"]
                    update_frame["boxes_lidar"]=PV_RCNN_frame["boxes_lidar"]
                    update_frame["metadata"]=PV_RCNN_frame["metadata"]
       
                    for frustum in result_frame["frustum"]:
                        if frustum["is_generated"] is True:
                            if frustum["label"]!="Sign":
                                update_frame["name"]=np.append(update_frame["name"],frustum["label"])
                                update_frame["score"]=np.append(update_frame["score"],frustum["score"].cpu().numpy())
                                update_frame["boxes_lidar"]=np.vstack((update_frame["boxes_lidar"],frustum["PVRCNN_Formed_Box"].astype("float32")))
                            
                    self.updated_result.append(update_frame)
                    break

if __name__ == "__main__":
    root = "./data/waymo/waymo_processed_data/"
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    infer=Inference(root,ckpt)
    infer.main()
    dir="/home/seongwon/SoftwareCapstone/data/waymo/waymo_processed_data/segment-1024360143612057520_3580_000_3600_000_with_camera_labels/segment-1024360143612057520_3580_000_3600_000_with_camera_labels.pkl"
