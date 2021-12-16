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
[Faster-RCNN(pytorch)](https://pytorch.org/) \
[Spconv](https://github.com/traveller59/spconv)\
[opend3D](http://www.open3d.org/)

## Build
```dotnetcli
python3 PVRCNN/setup.py build
```

## Waymo Dataset Preprocess
```dotnetcli
python3 PVRCNN/datasets/waymo/waymo_dataset.py --func create_waymo_infos --cfg_file PVRCNN/tools/cfgs/dataset_configs/waymo_dataset.yaml
```


## PVRCNN test
```dotnetcli
python3 test.py --cfg_file ./PVRCNN/tools/cfgs/waymo_models/pv_rcnn.yaml --batch_size 1 --ckpt [Cherckpoint Address]
```

## RUN MY Code
```dotnetcli
python3 inference.py
```

## Visualization
i를 원하는 프래임번호로 바꿔서 사용
```dotnetcli
python3 DrawMyResult(G)_PVRCNN(R) and GT(K).py
```

## Resource
3D Object Detection: PV-RCNN (mmlab)\
2D Object Detection: Faster R-CNN (pytorch)\
Waymo Google Dataset

## Project Explanation
실제로 많은 Lidar 기반 3D Object Detection에서 차량이 70프로가 넘는 경우가 많은데 비해 보행자에 대한 인식률은 60프로 미만으로 떨어진다.\
가장 검출 성능이 좋은 PV-RCNN을 직접 훈련 시켜서 결과를 한번 보았는데 가까이 있는 보행자들도 검출해내지 못했다.\
보행자를 검출하지 못한다면 자율주행사고 발생위험이 높아지고 인명의 피해가 발생할 수 있다는 것을 이야기한다.\
그래서 기존의 2D Image에서 2D Object Detection으로 예측한 결과와 PV-RCNN 결과를 합쳐서 이것을 해결해보려고한다.

## Structure
![CodeStructure](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/structure.png)
WaymoDataset.py: 전처리된 데이터셋에서 이미지 파일을 모델매니저로 로드 해주는 역할\
modelmanager.py:  PV-RCNN과 Faster R-CNN의 모델로 결과를 예측하는 역할\
fusion.py:  각각 예측값을 합쳐주는 역할 \
inferece.py: 그리고 실제 PV-RCNN과 나온 결과를 비교해주는 역할\
DrawMyResult(G)_PVRCNN(R) and GT(K).py: 이렇게 만들어진 결과를 보여주는 Visualization해주는 역할


## Result
![Result](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/result.png)

PVRCNN 에 비해서 보행자의 AP가 0.3%정도 증가했다.

자동차나 오토바이를 인식하는 부분에서는 기존의 방법과 큰 차이가 없었으나  보행자의 경우에서는 성능향상이 있었다. 
또한 새로 검출된 보행자들 중에는 자율주행차량과 가까이 있을 경우도 있었고, 이는 인사사고의 가능성을 조금이라도 더 줄였다는 것을 의미한다.

Waymo의 Sign class에 Ground Truth 해당 하지 않는 신호등도 검출을 할 수 있었다. 신호등을  검출했다는 것은 주변에 교차로가 있는지 횡단보도가 있는지 판단 할 수 있는 근거가 된다.

## Future Work
하지만 증가률이 낮은것은 사용한 Dataset이 Waymo인데 카메라가 후방에는 존재하지 않아서 전체를 커버하지 못하였기 때문이라고 생각한다.

따라서 추후에 360도를 다찍을 수있는 카메라를 사용하여 발전을 시킬 예정이다.