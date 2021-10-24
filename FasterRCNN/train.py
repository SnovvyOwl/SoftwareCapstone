from mmcv import Config
config_file = 'home/seongwon/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)
print(cfg)