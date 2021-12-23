# 3D Object Detection Using LiDAR and Camera

## Software Capstone Design [SWCON401]
### Department of Software Convergence, Kyung Hee Univ. 


Repository Owner: 
[SeongWon LEE](https://snovvyowl.github.io)\
Department of Mechanical Engineering, Kyung Hee Univesity

Advisor: 
Prof. HyoSeok Hwang\
Department of Software Convergence. 

## Resource
사용된 신경망 네트워크은 두개이다. 하나는 3D Obeject Detection
3D Object Detection: PV-RCNN (mmlab)\
2D Object Detection: Faster R-CNN (pytorch)\
Waymo Google Dataset
 @misc{waymo_open_dataset, title = {Waymo Open Dataset: An autonomous driving dataset}, website = {\url{https://www.waymo.com/open}}, year = {2019} }

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

## 주요 알고리즘 설명
### Calibration 
![Calibration](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/img.png)

LIDAR의 좌표계와 카메라의 이미지 좌표계가 일치 하지 않기 때문에 이를 해결하기 위해서는 Calibration을 진행해야한다.

![Calibrationeqn](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/calieqn.png)
LiDAR로 측정된 포인트들을 카메라의 extrinsic 행렬과 Intrinsic 행렬을 곱해서 이를 계산한다.


참고 문헌: [Barbara Frank, Cyrill Stachniss, Giorgio Grisetti, Kai Arras, Wolfram Burgard. Freiburg Univ. Lecture Note Robotics 2 Camera Calibration](http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-10-camera-calibration.pdf)

### Segmentation
segmetation을 위해 내가 유클리드 클러스팅을 직접구현하였으며 알고리즘은 아래와 같다.
![cluster](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/seg.png)
1)  입력 값으로 중심 포인트 좌표들과 Frustum을 넣어준다.
2)  중심좌표들은 Segmentation Set에 넣어준다.
3)  중심 좌표들로부터 특정거리 이하가 되면  새로 Segmentation set에 넣어준다.
4)  Segmentation set에 새로 넣어진 점을 기준으로 다시 계산을 해야 하므로 Queue에 추가 되는 인접 좌표를 넣어준다.
5)  한번 Segmentation된 결과에 포함된 포인트는 다시 계산하지 않도록 제외한다.
6)  Queue가 비어있지 않으면 Queue에서 포인트를 뽑아서 중심좌표로 선정하고 2~5를 반복한다.
7)  Queue가 비었다는 것은 추가된 점이 없다는 것으로 Segmentation 결과를 반환해준다. 

![cluster](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/center.png)
여기서 중심점을 계산한 방법은 다음같다. 

### PCA(Principal Component Analysis)
![eqn1](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/eqn1.png)
![eqn2](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/eqn2.png)
![eqn3](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/eqn3.png)
![eqn4](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/eqn4.png)


## Result & Conclusion
![Result](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/result.png)

PVRCNN 에 비해서 보행자의 AP가 0.3%정도 증가했다.

자동차나 오토바이를 인식하는 부분에서는 기존의 방법과 큰 차이가 없었으나  보행자의 경우에서는 성능향상이 있었다. 
또한 새로 검출된 보행자들 중에는 자율주행차량과 가까이 있을 경우도 있었고, 이는 인사사고의 가능성을 조금이라도 더 줄였다는 것을 의미한다.

Waymo의 Sign class에 Ground Truth 해당 하지 않는 신호등도 검출을 할 수 있었다. 신호등을  검출했다는 것은 주변에 교차로가 있는지 횡단보도가 있는지 판단 할 수 있는 근거가 된다.

결과사진은 WaymoDataset 라이선스 문제로 README.md에 따로 첨부하진 않겠다.


## Future Work
1. 하지만 증가률이 낮은것은 사용한 Dataset이 Waymo인데 카메라가 후방에는 존재하지 않아서 전체를 커버하지 못하였기 때문이라고 생각한다.

따라서 추후에 360도를 다찍을 수있는 카메라를 사용하여 발전을 시킬 예정이다.

2. 각각 다른 신경망 모델의 결과를 예측한것을 합쳐서 결과를 생성하는 2-Stage 방법으로 진행되어 계산 시간이 오래 걸렸다.
    따라서 많은 데이터셋에 대해 적용하지 못한 한계점을 가진다. 
    ->Faster R-CNN과 PV-RCNN의 구조가 비슷하므로 둘을 하나의 모델로 합칠수 있도록 하자.


# HOW TO BUILD AND RUN
사용된 라이브러리
[Dependency]\
[PV-RCNN](https://github.com/open-mmlab/OpenPCDet)\
[Faster-RCNN(pytorch)](https://pytorch.org/) \
[Spconv](https://github.com/traveller59/spconv)\
[opend3D](http://www.open3d.org/)


## PV-RCNN Build
### Build
```dotnetcli
python3 PVRCNN/setup.py build
```

### Waymo Dataset Preprocess
```dotnetcli
python3 PVRCNN/datasets/waymo/waymo_dataset.py --func create_waymo_infos --cfg_file PVRCNN/tools/cfgs/dataset_configs/waymo_dataset.yaml
```


### PVRCNN test
```dotnetcli
python3 test.py --cfg_file ./PVRCNN/tools/cfgs/waymo_models/pv_rcnn.yaml --batch_size 1 --ckpt [Cherckpoint Address]
```

## HOW TO RUN MY Code
### Inference
```dotnetcli
python3 inference.py
```
inference 파일을 실행하면 이미지와 포인트 클라우드를 받아서 ModelManager를 호출한다. 여기서 데이터 셋을 받아서 2D 예측과 3D 예측을 가지고 와서 이를 fusion을 하게 된다. 여기서 만들어진 결과로 기존의 PV-RCNN결과와 비교할수 있도록 수치화하는 코드다.


### Visualization
```if __name__ =="main":``` 
안에 있는 i라는 변수를 원하는 프래임번호로 바꿔서 코드를 돌리면 작동한다.
```dotnetcli
python3 DrawMyResult(G)_PVRCNN(R) and GT(K).py
```
결과는 검정상자는 Ground Truth, 빨간상자는 PV-RCNN 결과이며, 초록 박스가 새로 예측된 박스이다.  
