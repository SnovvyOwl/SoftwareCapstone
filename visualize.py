import pickle
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
with open("/home/seongwon/SoftwareCapstone/output/PVRCNN/tools/cfgs/waymo_models/pv_rcnn/default/eval/epoch_30/val/default/result.pkl",'rb') as f:
    groundtruth=pickle.load(f)
print(groundtruth)
print(len(groundtruth))
# with open('result.bin', 'wb') as k:
#     pickle.dump(groundtruth,k)