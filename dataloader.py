from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf
import numpy as np
import multiprocessing
import sys, os
import pickle
import pandas as pd

class WaymoSequenceLoader():
    def __init__(self, filepath, class_name):
        self.frames = []
        self.datasetpath = filepath

    def load_frames(self,FILENAME):
        '''
            Read TFrecord file and return one Frame
            Args:
            FILENAME:path of Waymo Data set (.tfrecord)
            frame: One Frame Record
        '''
        # FILENAME = '/home/seongwon/SoftwareCapstone/data/segment-13663273263251420352_1660_000_1680_000.tfrecord'
        sequence = tf.data.TFRecordDataset(FILENAME, compression_type='')
        for data in sequence:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            self.frames.append(frame)

    def get_range_image(self):
        for frame in self.frames:
            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            self.convert_range_image_to_point_cloud(range_images, camera_projections,range_image_top_pose)

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
        range_image_top_pose_tensor = tf.reshape(tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims)
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
    def save_lidar_data(self):
        # for bin in self.frames:
        
        return False

    def save_image(self):

        return False
    
    def save_box(self):

        return False
    
    def save_label(self):
        return False

class WaymoDataLoader(object):
    def __init__(self,rootpath, class_names,training=True):
        self.root=rootpath
        

        



if __name__=="__main__":
    seq1=WaymoSequenceLoader()