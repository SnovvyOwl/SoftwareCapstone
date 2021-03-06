import os
import pickle
import numpy as np
from ...utils import common_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2

from pathlib import Path
from PIL import Image
try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
#################################################################################0
##### WRITTEN BY SNOWYOWL
def save_images(frame,sequence_name,cur_save_dir,cnt):
    # 'FRONT_INTRINSIC', 'FRONT_EXTRINSIC', 'FRONT_WIDTH', 'FRONT_HEIGHT', 'FRONT_ROLLING_SHUTTER_DIRECTION' 'FRONT_IMAGE', 'FRONT_SDC_VELOCITY', 'FRONT_POSE', 'FRONT_POSE_TIMESTAMP', 'FRONT_ROLLING_SHUTTER_DURATION', 'FRONT_CAMERA_TRIGGER_TIME', 'FRONT_CAMERA_READOUT_DONE_TIME',
    # FRONT_CAM_PROJ_FIRST_RETURN', 'FRONT_CAM_PROJ_SECOND_RETURN',
    

    frame=frame_utils.convert_frame_to_dict(frame)
    camera_name=[ "FRONT_IMAGE", "FRONT_LEFT_IMAGE","FRONT_RIGHT_IMAGE", "SIDE_LEFT_IMAGE","SIDE_RIGHT_IMAGE"]
    filename=[]
    path=[]
    p=[]
    for camera_num in camera_name: #프레임당이미지 다섯개
        # camera_img.save(str(cur_save_dir)+"/"+sequence_name+"_"+camera_num+('_%04d.jpg'%cnt))
        path=str(cur_save_dir)+"/"+sequence_name+"_"+camera_num+('_%04d.jpg'%cnt)
        filename.append(str(sequence_name+"_"+camera_num+('_%04d.jpg'%cnt)))
        camera_img=Image.fromarray(frame[camera_num])
        camera_img.save(path)
    return  filename

def save_camera_calbration_parameter(frame,save_path):
    frame=frame_utils.convert_frame_to_dict(frame)
    intrinsics=[]
    extrinsics=[]
    camera_intrinsic_name=["FRONT_INTRINSIC","FRONT_LEFT_INTRINSIC", "FRONT_RIGHT_INTRINSIC","SIDE_LEFT_INTRINSIC" , "SIDE_RIGHT_INTRINSIC"]
    camera_extrinsic_name=["FRONT_EXTRINSIC","FRONT_LEFT_EXTRINSIC", "FRONT_RIGHT_EXTRINSIC","SIDE_LEFT_EXTRINSIC" , "SIDE_RIGHT_EXTRINSIC"]
    for camera_num in camera_intrinsic_name: #프레임당이미지 다섯개
        intrinsics.append(frame[camera_num])
    for camera_num in camera_extrinsic_name: #프레임당이미지 다섯개
        extrinsics.append(frame[camera_num])
    np.save(str(save_path)+"/extrinsic",extrinsics)
    np.save(str(save_path)+"/intrinsic",intrinsics)

def generate_camera_labels(frame,filename):
    camera=[]
    for labels in frame.projected_lidar_labels:
        info=make_label(labels,filename)
        camera.append(info)
 
       
    return camera[0],camera[1], camera[2],camera[3],camera[4] 
    
def make_label(labels,filename):
    anno=make_annotation(labels)
    info={}
    info['ann']=anno
    info['weight']=1920
    if labels.name in [1,2,3]:
        info['height']=1280
        if labels.name==1:
            info['filename']=filename[0]
        elif labels.name==2:
            info['filename']=filename[1]
        elif labels.name==3:
            info['filename']=filename[2]   
        
    else:
        info['height']=886
        if labels.name==4:
            info['filename']=filename[3]
        elif labels.name==5:
            info['filename']=filename[4]
    # queue.put(info)
    return info

def make_annotation(labels):
    types=[]
    boxes=[]
    ann={}
    for label in labels.labels:
        # print(label)
        box, cls_type =make_Bbox(label)
        boxes.append(box)
        types.append(cls_type)
    
    ann['bboxes']=np.array(boxes)
    ann['labels']=np.array(types)
    return ann

def make_Bbox(label):
    box2d=np.array([label.box.center_x - label.box.length / 2,label.box.center_y - label.box.width / 2,label.box.center_x + label.box.length / 2,label.box.center_y + label.box.width / 2])
    return box2d, WAYMO_CLASSES[label.type]

########################################################################################
#**************************************************************************************#
########################################################################################


# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.
def generate_labels(frame):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels
    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)

    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis]],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.
    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_NLZ = range_image_tensor[..., 3]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.where(range_image_mask))
        points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
        points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
        points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        points_NLZ.append(points_NLZ_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def save_lidar_points(frame, cur_save_path):
    range_images, camera_projections, range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = \
        convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_points_of_each_lidar


def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)
    ########################################################################################
    #************INCLUDE by Seongwon LEE***************************************************#
    sequence_camera=[]
    cur_img_dir=cur_save_dir/"img"
    cur_img_dir.mkdir(parents=True, exist_ok=True)
    pkl_img_file = cur_img_dir / ('camera_%s.pkl' % sequence_name)
    ########################################################################################
    sequence_infos = []
    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        print('Skip sequence since it has been processed before: %s' % pkl_file)
        return sequence_infos

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        ########################################################################################
        #******Camera Parameter Save      INCLUDE by Seongwon LEE******************************#
        save_camera_calbration_parameter(frame,cur_img_dir)
        ########################################################################################

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose
        ########################################################################################
        #******save image                 INCLUDE by Seongwon LEE******************************#
        filename=save_images(frame,sequence_name,cur_img_dir,cnt)
        ########################################################################################
        if has_label:
            annotations = generate_labels(frame)
            info['annos'] = annotations
            ########################################################################################
            #******generate_camera_bbox                 INCLUDE by Seongwon LEE******************************#
            FRONT_camera, FRONT_LEFT_camera, FRONT_RIGHT_camera, SIDE_LEFT_camera, SIDE_RIGHT_camera = generate_camera_labels(frame,filename)
        sequence_camera.append(FRONT_camera)
        sequence_camera.append(FRONT_LEFT_camera)
        sequence_camera.append(FRONT_RIGHT_camera)
        sequence_camera.append(SIDE_LEFT_camera)
        sequence_camera.append(SIDE_RIGHT_camera)
        #######################################################################################################################
        num_points_of_each_lidar = save_lidar_points(frame, cur_save_dir / ('%04d.npy' % cnt))
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)
    ###SAVE IMG SEQUCE#####
    with open(pkl_img_file,"wb") as file:
        pickle.dump(sequence_camera,file)
    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos