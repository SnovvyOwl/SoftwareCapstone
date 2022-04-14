from nuscenes import NuScenes
from nuscenes.scripts.export_kitti import KittiConverter
import sys, os
li=os.listdir("/home/seongwon/SoftwareCapstone/data/kitti/training/label_2")
f=open("/home/seongwon/SoftwareCapstone/data/kitti/ImageSets/train.txt",'w')
li2=os.listdir("/home/seongwon/SoftwareCapstone/data/kitti/testing/image_2")
f2=open("/home/seongwon/SoftwareCapstone/data/kitti/ImageSets/val.txt",'w')
for l in li:
    f.write(l[:-4]+"\n")
f.close()

for l in li2:
    print(l[:-4])
    f2.write(l[:-4]+"\n")
f2.close()
# kitti=KittiConverter("./nusc_kitti",root="/home/seongwon/SoftwareCapstone/data/nuscenes/v1.0-mini/",split="mini_val")
# kitti.nuscenes_gt_to_kitti()
