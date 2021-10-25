from re import S
from mmcv import image
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import numpy as np
import cv2
from mmdet.apis.inference import show_result_pyplot
from PIL import Image
import matplotlib.pyplot as plt
from numpy.core.records import array

labels_to_names_seq = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                       11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                       21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                       31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                       41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                       51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'sofa', 58: 'pottedplant', 59: 'bed', 60: 'diningtable',
                       61: 'toilet', 62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
                       71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


def get_detected_img(model, img_array,  score_threshold=0.3, is_print=True):
    # 인자로 들어온 image_array를 복사.
    draw_img = img_array.copy()
    bbox_color = (0, 255, 0)
    text_color = (0, 0, 255)

    # model과 image array를 입력 인자로 inference detection 수행하고 결과를 results로 받음.
    # results는 80개의 2차원 array(shape=(오브젝트갯수, 5))를 가지는 list.
    results = inference_detector(model, img_array)

    # 80개의 array원소를 가지는 results 리스트를 loop를 돌면서 개별 2차원 array들을 추출하고 이를 기반으로 이미지 시각화
    # results 리스트의 위치 index가 바로 COCO 매핑된 Class id. 여기서는 result_ind가 class id
    # 개별 2차원 array에 오브젝트별 좌표와 class confidence score 값을 가짐.
    for result_ind, result in enumerate(results):
        # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값이 없으므로 다음 loop로 진행.
        if len(result) == 0:
            continue

        # 2차원 array에서 5번째 컬럼에 해당하는 값이 score threshold이며 이 값이 함수 인자로 들어온 score_threshold 보다 낮은 경우는 제외.
        result_filtered = result[np.where(result[:, 4] > score_threshold)]

        # 해당 클래스 별로 Detect된 여러개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수만큼 iteration해서 개별 오브젝트의 좌표값 추출.
        for i in range(len(result_filtered)):
            # 좌상단, 우하단 좌표 추출.
            left = int(result_filtered[i, 0])
            top = int(result_filtered[i, 1])
            right = int(result_filtered[i, 2])
            bottom = int(result_filtered[i, 3])
            caption = "{}: {:.4f}".format(
                labels_to_names_seq[result_ind], result_filtered[i, 4])
            cv2.rectangle(draw_img, (left, top), (right, bottom),
                          color=bbox_color, thickness=2)
            cv2.putText(draw_img, caption, (int(left), int(top - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, text_color, 1)
            if is_print:
                print(caption)
        results=np.array(results)
        return draw_img, results


if __name__ == "__main__":
    # Choose to use a config and initialize the detector
    config = './FasterRCNN/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = './FasterRCNN/checkpoints/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')
    rootdir = "./data/waymo/waymo_processed_data/"
    sequence_list = os.listdir(rootdir)
    for sequence in sequence_list:
        image_list = os.listdir(
            "./data/waymo/waymo_processed_data/"+sequence+"/img/")
        for file in image_list:
            if file[-1] == "g":
                img_name = "./data/waymo/waymo_processed_data/"+sequence+"/img/"+file
                img = cv2.imread(img_name)
                detected ,result = get_detected_img(model, img)
                detected=Image.fromarray(detected)
                np.save("./FasterRCNN/output/"+file[0:-4]+".npy",result)
                detected.save("./FasterRCNN/output/"+file[0:-4]+".jpg")
