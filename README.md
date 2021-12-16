# 3D Object Detection Using LiDAR and Camera

## Software Capstone Design [SWCON401]
### Department of Software Convergence, Kyung Hee Univ. 


Repository Owner: 
[SeongWon LEE](https://snovvyowl.github.io)\
Department of Mechanical Engineering, Kyung Hee Univesity

Advisor: 
Prof. HyoSeok Hwang\
Department of Software Convergence. 

[Dependency]\
[PV-RCNN](https://github.com/open-mmlab/OpenPCDet)\
[Faster-RCNN](https://github.com/open-mmlab/mmdetection)
[Spconv](https://github.com/traveller59/spconv)

## Build
python3 PVRCNN/setup.py build

## Waymo Dataset Preprocess
python3 PVRCNN/datasets/waymo/waymo_dataset.py --func create_waymo_infos --cfg_file PVRCNN/tools/cfgs/dataset_configs/waymo_dataset.yaml

## PVRCNN test
python3 test.py --cfg_file ./PVRCNN/tools/cfgs/waymo_models/pv_rcnn.yaml --batch_size 1 --ckpt [Cherckpoint Address]

## Resource
3D Object Detection: PV-RCNN (mmlab)
2D Object Detection: Faster R-CNN (pytorch)
Waymo Google Dataset

## Project Explanation


## Structure
![CodeStructure](https://github.com/SnovvyOwl/Transparent_DepthEstimation/blob/main/DevelopmentLogs/codestruct.png)


## Result
![Result](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/result.png)

PVRCNN 에 비해서 AP가 0.3%정도 증가했다.

자동차나 오토바이를 인식하는 부분에서는 기존의 방법과 큰 차이가 없었으나  보행자의 경우에서는 성능향상이 있었다. 
또한 새로 검출된 보행자들 중에는 자율주행차량과 가까이 있을 경우도 있었고, 이는 인사사고의 가능성을 조금이라도 더 줄였다는 것을 의미한다.

Waymo의 Sign class에 Ground Truth 해당 하지 않는 신호등도 검출을 할 수 있었다. 신호등을  검출했다는 것은 주변에 교차로가 있는지 횡단보도가 있는지 판단 할 수 있는 근거가 된다.

## Future Work
하지만 증가률이 낮은것은 사용한 Dataset이 Waymo인데 카메라가 후방에는 존재하지 않아서 전체를 커버하지 못하였기 때문이라고 생각한다.

따라서 추후에 360도를 다찍을 수있는 카메라를 사용하여 발전을 시킬 예정이다.