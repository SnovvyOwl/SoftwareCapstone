import argparse
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
from kornia.geometry.bbox import validate_bbox
import tqdm
import time
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

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


def cocol2waymo(label):
    if label == 1:
        return WAYMO_CLASSES[2]
    elif label in [3, 5, 6, 7, 8, 9]:
        return WAYMO_CLASSES[1]
    elif label in [2, 4]:
        return WAYMO_CLASSES[4]
    elif label == 10:
        return WAYMO_CLASSES[3]
    elif label not in range(1, 11):
        return WAYMO_CLASSES[0]


class ModelManager(object):
    def __init__(self, root, PCckpt):
        self.sequence = None
        self.cfg = EasyDict()
        self.root = root
        self.imgloaded = None
        self.PCloaded = None

        self.args, self.cfg = self.parse_config(PCckpt)
        self.PVRCNN_model = self.build_PVRCNN_Model()
        self.FASTERRCNN_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.FASTERRCNN_model.cuda()
        self.FASTERRCNN_model.eval()
        self.logger

    def parse_config(self, ckptdir):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default="PVRCNN/tools/cfgs/waymo_models/pv_rcnn.yaml",
                            help='specify the config for training')

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
        parser.add_argument('--eval_all', action='store_true', default=False,
                            help='whether to evaluate all checkpoints')
        parser.add_argument('--ckpt_dir', type=str, default=ckptdir,
                            help='specify a ckpt directory to be evaluated if needed')
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
        self.logger = common_utils.create_logger(None, rank=cfg.LOCAL_RANK)
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
        self.test_set, self.test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=self.args.batch_size,
            dist=dist_test, workers=self.args.workers, logger=self.logger, training=False
        )
        model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.test_set)
        model.load_params_from_file(filename=self.args.ckpt, logger=self.logger, to_cpu=dist_test)
        model.cuda()
        with torch.no_grad():
            dataset = self.test_loader.dataset
            class_names = dataset.class_names
            det_annos = []
            model.eval()
            return model

    def val(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        final_output_dir = None
        save_to_file = False
        metric = {
            'gt_num': 0,
        }
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] = 0
            metric['recall_rcnn_%s' % str(cur_thresh)] = 0
        dataset = self.test_loader.dataset
        class_names = dataset.class_names
        det_annos = []
        img_annos = []
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='eval', dynamic_ncols=True)
        start_time = time.time()
        for i, batch_dict in enumerate(self.test_loader):
            idx = int(batch_dict["frame_id"][0][-3:])
         
            sequence_id = batch_dict["frame_id"][0][:-4]
            load_data_to_gpu(batch_dict)
           
            if sequence_id != self.sequence:
                self.sequence = sequence_id
                self.imgloaded = Waymo2DLoader(self.root, self.sequence)
                # self.PCloaded=Waymo3DLoader(self.root,self.sequnce)

            with torch.no_grad():
                pred_dicts, ret_dict = self.PVRCNN_model(batch_dict)  # 5개 당 하나씩 나옴
            disp_dict = {}
            eval_utils.statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if save_to_file else None
            )
            img_pred = {}
            img_pred["extrinsic"] = self.imgloaded.extrinsic
            img_pred["intrinsic"] = self.imgloaded.intrinsic
            imgs, targets = self.imgloaded.__getitem__(idx)
            img_pred["imgs"] = imgs
            img_pred["anno"] = []
            img_pred["frame_id"] = batch_dict["frame_id"]
            img_pred["image_id"]=[]
            for i, img in enumerate(imgs):
                img = transform(img).cuda()
                pred_one_img = self.pred_2Dbox(img)  # 2d BOX anNOTATION. FOR 1 Image
                img_pred["anno"].append(pred_one_img)
                img_pred["image_id"].append(targets[i]["image_id"])
            img_annos.append(img_pred)
            det_annos += annos
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
        progress_bar.close()
        self.logger.info('****************Evaluation done.*****************')
        return det_annos, img_annos

    def pred_2Dbox(self, img):
        pred_class = []
        with torch.no_grad():
            pred = self.FASTERRCNN_model([img])
        pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].cpu().numpy())]
        for i in list(pred[0]['labels'].cpu().numpy()):
            pred_class.append(cocol2waymo(i))
        pred[0]['labels'] = pred_class
        pred[0]['boxes']=pred_boxes
        return pred[0]


if __name__ == "__main__":
    import pickle
    root = "./data/waymo/waymo_processed_data/"
    ckpt = "./checkpoints/checkpoint_epoch_30.pth"
    test = ModelManager(root, ckpt)
    a,b=test.val()
    with open( "anno3d.pkl", 'wb') as f:
        pickle.dump(a,f)
    with open("anno2d.pkl",'wb') as f:
        pickle.dump(b,f)
