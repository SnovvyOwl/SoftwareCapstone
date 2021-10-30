import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
from kornia.geometry.bbox import validate_bbox

import numpy as np
import torch
from tensorboardX import SummaryWriter

from PVRCNN.tools.eval_utils import eval_utils
from PVRCNN.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from PVRCNN.datasets import build_dataloader
from PVRCNN.models import build_network, load_data_to_gpu
from PVRCNN.utils import common_utils
from WaymoDataset import *
from easydict import EasyDict

def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )

class ValidationEachScene(object):
    def __init__(self,root,PCckpt):
        self.sequnce=None
        self.cfg=EasyDict()
        self.root=root
        self.imgloaded=None
        self.PCloaded=None
       
        self.args,self.cfg=self.parse_config(PCckpt)
        self.PVRCNN_model=self.build_PVRCNN_Model()
        self.FASTERRCNN_model=None
        
    def parse_config(self,ckptdir):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default="PVRCNN/tools/cfgs/waymo_models/pv_rcnn.yaml", help='specify the config for training')

        parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
        parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
        parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
        parser.add_argument('--ckpt', type=str, default=ckptdir, help='checkpoint to start from')
        parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
        parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
        parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

        parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
        parser.add_argument('--start_epoch', type=int, default=0, help='')
        parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
        parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
        parser.add_argument('--ckpt_dir', type=str, default=ckptdir, help='specify a ckpt directory to be evaluated if needed')
        parser.add_argument('--save_to_file', action='store_true', default=False, help='')

        args = parser.parse_args()

        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.TAG = Path(args.cfg_file).stem
        cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

        np.random.seed(1024)

        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs, cfg)
        return args, cfg
    
    def build_PVRCNN_Model(self):
        logger = common_utils.create_logger(None, rank=cfg.LOCAL_RANK)
        if self.args.launcher == 'none':
            dist_test = False
            total_gpus = 1
        else:
            total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % self.args.launcher)(
            self.args.tcp_port, self.args.local_rank, backend='nccl'
        )
        dist_test = True

        if self.args.batch_size is None:
            self.args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        else:
            assert self.args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
            self.args.batch_size = self.args.batch_size // total_gpus
        test_set, self.test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=self.args.batch_size,
            dist=dist_test, workers=self.args.workers, logger=logger, training=False
        )
        model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=test_set)
        print(self.args.ckpt)
        model.load_params_from_file(filename=self.args.ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()
        with torch.no_grad():
            dataset = self.test_loader.dataset
            class_names = dataset.class_names
            det_annos = []
            model.eval()
            return model

    def val(self):
        for i, batch_dict in enumerate(self.test_loader):
            idx=int(batch_dict["frame_id"][0][-3:])
            sequnce_id=batch_dict["frame_id"][0][:-4]
            load_data_to_gpu(batch_dict)
            if sequnce_id is not self.sequnce:
                self.sequnce=sequnce_id
                self.imgloaded=Waymo2DLoader(self.root,self.sequnce)
                self.PCloaded=Waymo3DLoader(self.root,self.sequnce)
            with torch.no_grad():
                pred_dicts, ret_dict = self.PVRCNN_model(batch_dict) # 5개 당 하나씩 나옴
            disp_dict = {}
            
        
        

if __name__=="__main__":
    root="./data/waymo/waymo_processed_data/"
    sequece='segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt="/home/seongwon/SoftwareCapstone/checkpoints/checkpoint_epoch_30.pth"
    test=ValidationEachScene(root,ckpt)
    test.val()