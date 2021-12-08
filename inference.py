import pickle
import numpy as np
import torch
import open3d as o3d
from fusion import Fusion
from PVRCNN.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
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
        # self.fusion=Fusion(root,ckptdir)
    
    def load_ground_truth_data(self):
        dirpath="./data/waymo/waymo_infos_val.pkl"
        with open(dirpath,"rb") as f:
            self.gt_data=pickle.load(f)

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
                    # Calcluate IoU For One Frame
                    sign_idx=np.where(gt_frame["annos"]["name"]=="Sign")
                    iou_mat=boxes_iou3d_gpu(torch.tensor(gt_frame["annos"]["gt_boxes_lidar"].astype("float32")).cuda(),torch.tensor(frame["boxes_lidar"]).cuda())
                    iou_mat=iou_mat.cpu().numpy()
                    iou_mat=np.delete(iou_mat,sign_idx,axis=0)
                    gt_name=np.delete(gt_frame["annos"]["name"],sign_idx,axis=0)
                    self.match_correction(gt_name,frame["name"],iou_mat,0.5)
                    break
    
    def match_correction(self,gt_name,frame_name,iou_mat,thresh=0.5):
        TP=0
        FP=0
        match_result=np.array(np.where(iou_mat>thresh)).T
        for match in match_result:
            if gt_name[match[0]]==frame_name[match[1]]:
                TP+=1
                print(iou_mat[match[0],match[1]])
            else:
                FP+=1
        all_detection=iou_mat.shape[1]
        all_gt=len(gt_name)
        print(float(TP)/float(all_detection))
        print(float(TP)/float(all_gt))
        return all_gt,all_detection,TP,FP
    
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
