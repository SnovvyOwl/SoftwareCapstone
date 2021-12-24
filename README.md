# 3D Object Detection Using LiDAR and Camera

## Software Capstone Design [SWCON401]
### Department of Software Convergence, Kyung Hee Univ. 


Repository Owner: 
[SeongWon LEE](https://snovvyowl.github.io)\
Department of Mechanical Engineering, Kyung Hee Univesity

Advisor: 
Prof. HyoSeok Hwang\
Department of Software Convergence. 

## Problem
자율 주행 자동차는 자동차 스스로 승객의 조작이 없이 운행가능한 자동차이다.\
하지만 자동차가 스스로 운행을 하기 위해서는 주변환경을 인지하는 능력이 있어야 한다. \
따라서 주변환경을 인지하기 위해서 여러 센서를 사용하게 되는데 그중 가장 대표적으로 Lidar를 사용한다.\
실제로 많은 Lidar 기반 3D Object Detection에서 차량이 70프로가 넘는 경우가 많은데 비해 보행자에 대한 인식률은 60프로 미만으로 떨어진다.\
![problem](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/problem.png)

가장 검출 성능이 좋은 PV-RCNN을 직접 훈련 시켜서 결과를 한번 보았는데 가까이 있는 보행자들도 검출해내지 못했다.\
보행자를 검출하지 못한다면 자율주행사고 발생위험이 높아지고 인명의 피해가 발생할 수 있다는 것을 이야기한다.

포인트클라우드를 이용한 물체 검출보다 이미지를 사용한 검출은 검출률이 높다. \
하지만 이미지는 물체와의 거리 인식의 정확도가 라이다보다 떨어져서 라이다의 문제점과 카메라의 문제점이 상호 보완이 되기 때문에 이 두가지 센서를 퓨전하려고 한다.\
즉, 2D Image에서 2D Object Detection으로 예측한 결과와 Lidar의 포인트클라우드를 이용하는 PV-RCNN 결과를 합쳐서 이것을 해결해보려고한다.

## Resource
사용된 신경망 네트워크은 두개이며 사용된 데이터 셋은 Waymo Dataset이다.

### 3D Object Detection: PV-RCNN 
(S. Shi et al., "PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection,“
 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 10526-10535, doi: 10.1109/CVPR42600.2020.01054.)

![PVRCNN](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/PVRCNN.png)

PV-RCNN은 Faster R-CNN의 구조를 본딴 구조를 가지고 있으며 RoI를 만들어주는 보라색박스, feature volume에 해당하는 초록색 박스 실제 classification과 Box refinement를 하는 파란 박스로 구현되어 있다. 마지막 파란박스는 PointNet과 구조가 비슷하다. 여기서 ROI는 Voxel 기반으로 CNN을 통과시켜 Bird Eye View로 찾는것이고 
feauture volume을 만드는데 사용된 VSA 모듈이 특징이다.

### 2D Object Detection: Faster R-CNN
(Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards real-time object detection with region proposal networks.Advances in neural information processing systems,28, 91-99.)

![FasterRCNN](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/fasterRCNN.png)

Faster R-CNN은 R-CNN 계열 2-stage object detection modeld이며 R-CNN의 ROI각각 CNN 네트워크를 통과시는 것을 개량하여 하나의 이미지를 CNN에 통과시킨후 RoI pooling을하는 Fast R-CNN을 거쳐 RoI 자체도 네트워크를 만들어 결과를 만드는 것으로 개량된 네트워크이다.

### Waymo Google Dataset
( P. Sun et al., "Scalability in Perception for Autonomous Driving: Waymo Open Dataset," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 2443-2451, doi: 10.1109/CVPR42600.2020.00252.)
 @misc{waymo_open_dataset, title = {Waymo Open Dataset: An autonomous driving dataset}, website = {\url{https://www.waymo.com/open}}, year = {2019} }

![WAYMO](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/waymo.png)

Waymo는 4개의 Short Range Lidar와 1개의 Mid Range LiDAR를 사용하는 데이터 셋이며 카메라는 5개가 사용된다. 
후방에는 이미지 센서가 없지만 프레임당 평균 포인트수가 많아서 이를 선정했다.

## Structure
![CodeStructure](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/structure.png)
WaymoDataset.py: 전처리된 데이터셋에서 이미지 파일을 모델매니저로 로드 해주는 역할\
modelmanager.py:  PV-RCNN과 Faster R-CNN의 모델로 결과를 예측하는 역할\
fusion.py:  각각 예측값을 합쳐주는 역할 \
inferece.py: 그리고 실제 PV-RCNN과 나온 결과를 비교해주는 역할\
DrawMyResult(G)_PVRCNN(R) and GT(K).py: 이렇게 만들어진 결과를 보여주는 Visualization해주는 역할

## 주요 알고리즘 설명
### Frustum?
프러스텀은  이미지 픽셀에 찍힌 빛이 있을 수 있는 공간이다. 이미지에 기록된 빛은 어디서 왔는지 방향만 알수있고 이는 사각뿔에서 꼭대기가 짤린형태라하여 절두체라고한다. 
![Frustum](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/frustum.png)

참고문헌: Y. Wei, S. Su, J. Lu and J. Zhou, "FGR: Frustum-Aware Geometric Reasoning for Weakly Supervised 3D Vehicle Detection," 
2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 4348-4354, doi: 10.1109/ICRA48506.2021.9561245.

### Calibration 
![Calibration](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/img.png)

LIDAR의 좌표계와 카메라의 이미지 좌표계가 일치 하지 않기 때문에 이를 해결하기 위해서는 Calibration을 진행해야한다.

![Calibrationeqn](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/calieqn.png)

LiDAR로 측정된 포인트들을 카메라의 extrinsic 행렬과 Intrinsic 행렬을 곱해서 이를 계산한다.

#### 이미지별 Frustum
![CaliRESULT](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/calibration.png)

#### 2D BOX 별 Frustum
![frustumRESULT](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/frustum2.png)

참고 문헌: [Barbara Frank, Cyrill Stachniss, Giorgio Grisetti, Kai Arras, Wolfram Burgard. Freiburg Univ. Lecture Note Robotics 2 Camera Calibration](http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-10-camera-calibration.pdf)

### Segmentation
그렇다면 위의 2D박스별 Frustum에서 실제 물체와 물체가 아닌 것을 구별(Segmentation)해야한다. 
segmetation을 위해 내가 유클리드 클러스팅을 직접구현하였으며 알고리즘은 아래와 같다.

![cluster](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/seg.png)

1)  입력 값으로 중심 포인트 좌표들과 Frustum을 넣어준다.
2)  중심좌표들은 Segmentation Set에 넣어준다.
3)  중심 좌표들로부터 특정거리 이하가 되면  새로 Segmentation set에 넣어준다.
4)  Segmentation set에 새로 넣어진 점을 기준으로 다시 계산을 해야 하므로 Queue에 추가 되는 인접 좌표를 넣어준다.
5)  한번 Segmentation된 결과에 포함된 포인트는 다시 계산하지 않도록 제외한다.
6)  Queue가 비어있지 않으면 Queue에서 포인트를 뽑아서 중심좌표로 선정하고 2~5를 반복한다.
7)  Queue가 비었다는 것은 추가된 점이 없다는 것으로 Segmentation 결과를 반환해준다. 



여기서 중심점을 계산한 방법은 다음같다. 

![center](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/center.png)

1) Faster RCNN으로 생성된 박스크기에 1%크기의 작은 박스를 생성한다. 
2)작은 박스에 포함되는 Point Cloud를 원점으로부터 가장 멀리 있는 점과 가장 가까운 점을 가지고 Segmentation을 진행한다.
3)두 결과에서 포함하는 포인트수가 다를 경우 작은 쪽을 지운다.
4) 두 결과의 크기가 같고 둘의 합이 원래 센터 박스일 경우 조금 더 큰 센터박스를 생성하여 한번 더 진행한다.
5)만약 둘의 크기가 센터박스와 같을 경우 센터박스에 포함되는 모든 포인트를 센터포인트라고 판단한다.


### PCA(Principal Component Analysis)
3D Object Detection은 2D와 다르게 상자의 회전각도도 중요하다. 따라서 Segmentation 결과를 가지고 결과의 좌표축을 알 필요가 있다. 
따라서 이를 알기위해 PCA를 구현하여 박스를 만들었다.

![eqn1](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/eqn1.png)
![eqn2](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/eqn2.png)

공분산 행렬을 구한 후 이를 대각화한다.

![eqn4](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/eqn4.png)
![eqn3](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/eqn3.png)

참고문헌: H. Vceraraghavan, O. Masoud and N. Papanikolopoulos, "Vision-based monitoring of intersections," Proceedings. 
The IEEE 5th International Conference on Intelligent Transportation Systems, 2002, pp. 7-12, doi: 10.1109/ITSC.2002.1041180.

## Result & Conclusion
### 결과 사진 
빨간상자는 PV-R-CNN 결과이고, 검정상자는 Ground Truth 결과,초록 상자는 위의 방법으로 생성된 결과이다.

![RESULT_IMG1](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/res1.png)
![RESULT_IMG2](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/res2.png)
![RESULT_IMG3](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/res3.png)

왼쪽사진은 실제 이미지이고 오른쪽은 생성된 결과를 보여준다.
결과적으로 보행자를 추가 검출하는데에 성공했다.

이를 수치화한 결과이다.
![Result](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/result.png)

PVRCNN 에 비해서 보행자의 AP가 0.3%정도 증가했다.

자동차나 오토바이를 인식하는 부분에서는 기존의 방법과 큰 차이가 없었으나  보행자의 경우에서는 성능향상이 있었다. 
또한 새로 검출된 보행자들 중에는 자율주행차량과 가까이 있을 경우도 있었고, 이는 인사사고의 가능성을 조금이라도 더 줄였다는 것을 의미한다.

![RESULT_IMG4](https://github.com/SnovvyOwl/SoftwareCapstone/blob/main/doc/res4.png)
Waymo의 Sign class에 Ground Truth 해당 하지 않는 신호등도 검출을 할 수 있었다. 신호등을  검출했다는 것은 주변에 교차로가 있는지 횡단보도가 있는지 판단 할 수 있는 근거가 된다.


## Future Work
1. 하지만 증가률이 낮은것은 사용한 Dataset이 Waymo인데 카메라가 후방에는 존재하지 않아서 전체를 커버하지 못하였기 때문이라고 생각한다.

따라서 추후에 360도를 다찍을 수있는 카메라를 사용하여 발전을 시킬 예정이다.

2. 각각 다른 신경망 모델의 결과를 예측한것을 합쳐서 결과를 생성하는 2-Stage 방법으로 진행되어 계산 시간이 오래 걸렸다.
    따라서 많은 데이터셋에 대해 적용하지 못한 한계점을 가진다. 
    ->Faster R-CNN과 PV-RCNN의 구조가 비슷하므로 둘을 하나의 모델로 합칠수 있도록 하자.


# HOW TO BUILD AND RUN

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
