from mmdet.apis import init_detector
import os
from sequnce_eval import FastRCNN_sequence

class FastRCNN_eval(object):
    def __init__(self,config,checkpoint,rootdir):
        self.cfg=config
        self.root=rootdir
        self.sequence_list = os.listdir(rootdir)
        # initialize the detector
        self.model=init_detector(config, checkpoint, device='cuda:0')

    def save_result(self,outputdir,saveimage=True,GT_compare=True):
        for sequence in self.sequence_list:
            seq=FastRCNN_sequence(self.root,sequence,outputdir,saveimage,GT_compare)
            seq.save(self.model)

if __name__ == "__main__":
    # Choose to use a config and initialize the detector
    config = './FasterRCNN/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = './FasterRCNN/checkpoints/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth'
    rootdir = "./PV_R-CNN/data/waymo/waymo_processed_data/"
    outputdir="./FasterRCNN/output/"
    eval=FastRCNN_eval(config,checkpoint,rootdir)
    eval.save_result(outputdir)