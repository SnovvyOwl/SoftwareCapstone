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


## Structure
![CodeStructure](https://github.com/SnovvyOwl/Transparent_DepthEstimation/blob/main/DevelopmentLogs/codestruct.png)


## Result
![Result](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/result.png)

PVRCNN 에 비해서 AP가 0.3%정도 증가했다.


![Myresult_Generated](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/segboxresult.png)

이 사진과 같이 사람과 차량의 거리가 매우 가까움에도 기존의 PV-RCNN은 검출하지 못하였고 
카메라와의 센서퓨전을 통한 나의 결과로 이 프레임에서 2명의 사람을 추가로 검출할수 있었다.

이는 자율주행의 인사사고 확률을 조금이라도 낮출수 있다.

## Future Work
하지만 증가률이 낮은것은 사용한 Dataset이 Waymo인데 카메라가 후방에는 존재하지 않아서 전체를 커버하지 못하였기 때문이라고 생각한다.

따라서 추후에 360도를 다찍을 수있는 카메라를 사용하여 발전을 시킬 예정이다.