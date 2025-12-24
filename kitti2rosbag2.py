#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

try:
    import pykitti
except ImportError as e:
    print('Could not load module \'pykitti\'. Please run `pip install pykitti`')
    sys.exit(1)

import os
import cv2
import progressbar
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

# ROS2 imports
from rclpy.serialization import serialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from builtin_interfaces.msg import Time


def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion using scipy."""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    quat = r.as_quat()  # returns [x, y, z, w]
    return quat


def quaternion_from_matrix(matrix):
    """Convert rotation matrix to quaternion using scipy."""
    r = R.from_matrix(matrix[0:3, 0:3])
    quat = r.as_quat()  # returns [x, y, z, w]
    return quat


def stamp_to_ros_time(timestamp_sec):
    """Convert timestamp in seconds to ROS2 Time message."""
    time_msg = Time()
    time_msg.sec = int(timestamp_sec)
    time_msg.nanosec = int((timestamp_sec - int(timestamp_sec)) * 1e9)
    return time_msg


def save_imu_data(bag, kitti, imu_frame_id, topic, time_offset=0.0):
    print("Exporting IMU")
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        q = quaternion_from_euler(oxts.packet.roll, oxts.packet.pitch, oxts.packet.yaw)
        imu = Imu()
        imu.header.frame_id = imu_frame_id
        timestamp_sec = float(timestamp.strftime("%s.%f")) + time_offset
        imu.header.stamp = stamp_to_ros_time(timestamp_sec)
        imu.orientation.x = q[0]
        imu.orientation.y = q[1]
        imu.orientation.z = q[2]
        imu.orientation.w = q[3]
        imu.linear_acceleration.x = oxts.packet.af
        imu.linear_acceleration.y = oxts.packet.al
        imu.linear_acceleration.z = oxts.packet.au
        imu.angular_velocity.x = oxts.packet.wf
        imu.angular_velocity.y = oxts.packet.wl
        imu.angular_velocity.z = oxts.packet.wu
        
        timestamp_ns = int(timestamp_sec * 1e9)
        bag.write(topic, serialize_message(imu), timestamp_ns)


def save_dynamic_tf(bag, kitti, kitti_type, initial_time, time_offset=0.0):
    print("Exporting time dependent transformations")
    if kitti_type.find("raw") != -1:
        for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
            tf_oxts_msg = TFMessage()
            tf_oxts_transform = TransformStamped()
            timestamp_sec = float(timestamp.strftime("%s.%f")) + time_offset
            tf_oxts_transform.header.stamp = stamp_to_ros_time(timestamp_sec)
            tf_oxts_transform.header.frame_id = 'world'
            tf_oxts_transform.child_frame_id = 'base_link'

            transform = (oxts.T_w_imu)
            t = transform[0:3, 3]
            q = quaternion_from_matrix(transform)
            oxts_tf = Transform()

            oxts_tf.translation.x = t[0]
            oxts_tf.translation.y = t[1]
            oxts_tf.translation.z = t[2]

            oxts_tf.rotation.x = q[0]
            oxts_tf.rotation.y = q[1]
            oxts_tf.rotation.z = q[2]
            oxts_tf.rotation.w = q[3]

            tf_oxts_transform.transform = oxts_tf
            tf_oxts_msg.transforms.append(tf_oxts_transform)

            timestamp_ns = int(timestamp_sec * 1e9)
            bag.write('/tf', serialize_message(tf_oxts_msg), timestamp_ns)

    elif kitti_type.find("odom") != -1:
        timestamps = [initial_time + x.total_seconds() for x in kitti.timestamps]
        for timestamp, tf_matrix in zip(timestamps, kitti.T_w_cam0):
            tf_msg = TFMessage()
            tf_stamped = TransformStamped()
            tf_stamped.header.stamp = stamp_to_ros_time(timestamp)
            tf_stamped.header.frame_id = 'world'
            tf_stamped.child_frame_id = 'camera_left'
            
            t = tf_matrix[0:3, 3]
            q = quaternion_from_matrix(tf_matrix)
            transform = Transform()

            transform.translation.x = t[0]
            transform.translation.y = t[1]
            transform.translation.z = t[2]

            transform.rotation.x = q[0]
            transform.rotation.y = q[1]
            transform.rotation.z = q[2]
            transform.rotation.w = q[3]

            tf_stamped.transform = transform
            tf_msg.transforms.append(tf_stamped)

            timestamp_ns = int(timestamp * 1e9)
            bag.write('/tf', serialize_message(tf_msg), timestamp_ns)


def save_camera_data(bag, kitti_type, kitti, util, bridge, camera, camera_frame_id, topic, initial_time, time_offset=0.0):
    print("Exporting camera {}".format(camera))
    if kitti_type.find("raw") != -1:
        camera_pad = '{0:02d}'.format(camera)
        image_dir = os.path.join(kitti.data_path, 'image_{}'.format(camera_pad))
        image_path = os.path.join(image_dir, 'data')
        image_filenames = sorted(os.listdir(image_path))
        with open(os.path.join(image_dir, 'timestamps.txt')) as f:
            image_datetimes = []
            for line in f.readlines():
                timestamp_str = line.strip()
                # Handle timestamps with more than 6 decimal places
                if '.' in timestamp_str:
                    base, fraction = timestamp_str.rsplit('.', 1)
                    # Truncate to 6 decimal places for microseconds
                    fraction = fraction[:6]
                    timestamp_str = base + '.' + fraction
                image_datetimes.append(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f'))
        
        calib = CameraInfo()
        calib.header.frame_id = camera_frame_id
        s_rect = util['S_rect_{}'.format(camera_pad)].tolist()
        calib.width = int(s_rect[0])
        calib.height = int(s_rect[1])
        calib.distortion_model = 'plumb_bob'
        calib.k = util['K_{}'.format(camera_pad)].flatten().tolist()
        calib.r = util['R_rect_{}'.format(camera_pad)].flatten().tolist()
        calib.d = util['D_{}'.format(camera_pad)].flatten().tolist()
        calib.p = util['P_rect_{}'.format(camera_pad)].flatten().tolist()
            
    elif kitti_type.find("odom") != -1:
        camera_pad = '{0:01d}'.format(camera)
        image_path = os.path.join(kitti.sequence_path, 'image_{}'.format(camera_pad))
        image_filenames = sorted(os.listdir(image_path))
        image_datetimes = [initial_time + x.total_seconds() for x in kitti.timestamps]
        
        calib = CameraInfo()
        calib.header.frame_id = camera_frame_id
        calib.p = util['P{}'.format(camera_pad)].flatten().tolist()
    
    iterable = list(zip(image_datetimes, image_filenames))
    bar = progressbar.ProgressBar(maxval=len(iterable))
    for dt, filename in bar(iterable):
        image_filename = os.path.join(image_path, filename)
        cv_image = cv2.imread(image_filename)
        calib.height, calib.width = cv_image.shape[:2]
        if camera in (0, 1):
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        encoding = "mono8" if camera in (0, 1) else "bgr8"
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
        image_message.header.frame_id = camera_frame_id
        
        if kitti_type.find("raw") != -1:
            timestamp_sec = float(datetime.strftime(dt, "%s.%f")) + time_offset
            image_message.header.stamp = stamp_to_ros_time(timestamp_sec)
            topic_ext = "/image_raw"
        elif kitti_type.find("odom") != -1:
            timestamp_sec = dt + time_offset
            image_message.header.stamp = stamp_to_ros_time(timestamp_sec)
            topic_ext = "/image_rect"
            
        calib.header.stamp = image_message.header.stamp
        timestamp_ns = int(timestamp_sec * 1e9)
        
        bag.write(topic + topic_ext, serialize_message(image_message), timestamp_ns)
        bag.write(topic + '/camera_info', serialize_message(calib), timestamp_ns)


def create_point_cloud2(header, points):
    """Create a PointCloud2 message from numpy array."""
    from sensor_msgs.msg import PointCloud2
    
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = points.shape[0]
    msg.is_dense = False
    msg.is_bigendian = False
    
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    
    msg.point_step = 16
    msg.row_step = msg.point_step * points.shape[0]
    msg.data = points.tobytes()
    
    return msg


def save_velo_data(bag, kitti, velo_frame_id, topic, time_offset=0.0):
    print("Exporting velodyne data")
    velo_path = os.path.join(kitti.data_path, 'velodyne_points')
    velo_data_dir = os.path.join(velo_path, 'data')
    velo_filenames = sorted(os.listdir(velo_data_dir))
    with open(os.path.join(velo_path, 'timestamps.txt')) as f:
        lines = f.readlines()
        velo_datetimes = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            # Handle timestamps with more than 6 decimal places
            if '.' in line:
                base, fraction = line.rsplit('.', 1)
                # Truncate to 6 decimal places for microseconds
                fraction = fraction[:6]
                line = base + '.' + fraction
            dt = datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f')
            velo_datetimes.append(dt)

    iterable = list(zip(velo_datetimes, velo_filenames))
    bar = progressbar.ProgressBar(maxval=len(iterable))
    for dt, filename in bar(iterable):
        if dt is None:
            continue

        velo_filename = os.path.join(velo_data_dir, filename)

        # read binary data
        scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)

        # create header
        header = Header()
        header.frame_id = velo_frame_id
        timestamp_sec = float(datetime.strftime(dt, "%s.%f")) + time_offset
        header.stamp = stamp_to_ros_time(timestamp_sec)

        # fill pcl msg
        pcl_msg = create_point_cloud2(header, scan)

        timestamp_ns = int(timestamp_sec * 1e9)
        bag.write(topic + '/pointcloud', serialize_message(pcl_msg), timestamp_ns)


def get_static_transform(from_frame_id, to_frame_id, transform):
    t = transform[0:3, 3]
    q = quaternion_from_matrix(transform)
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = from_frame_id
    tf_msg.child_frame_id = to_frame_id
    tf_msg.transform.translation.x = float(t[0])
    tf_msg.transform.translation.y = float(t[1])
    tf_msg.transform.translation.z = float(t[2])
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg


def inv(transform):
    """Invert rigid body transformation matrix"""
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1 * R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv
    return transform_inv


def save_static_transforms(bag, transforms, timestamps, time_offset=0.0):
    print("Exporting static transformations")
    tfm = TFMessage()
    for transform in transforms:
        t = get_static_transform(from_frame_id=transform[0], to_frame_id=transform[1], transform=transform[2])
        tfm.transforms.append(t)
    for timestamp in timestamps:
        timestamp_sec = float(timestamp.strftime("%s.%f")) + time_offset
        time = stamp_to_ros_time(timestamp_sec)
        for i in range(len(tfm.transforms)):
            tfm.transforms[i].header.stamp = time
        timestamp_ns = int(timestamp_sec * 1e9)
        bag.write('/tf_static', serialize_message(tfm), timestamp_ns)


def save_gps_fix_data(bag, kitti, gps_frame_id, topic, time_offset=0.0):
    print("Exporting GPS fix data")
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        navsatfix_msg = NavSatFix()
        navsatfix_msg.header.frame_id = gps_frame_id
        timestamp_sec = float(timestamp.strftime("%s.%f")) + time_offset
        navsatfix_msg.header.stamp = stamp_to_ros_time(timestamp_sec)
        navsatfix_msg.latitude = oxts.packet.lat
        navsatfix_msg.longitude = oxts.packet.lon
        navsatfix_msg.altitude = oxts.packet.alt
        navsatfix_msg.status.service = 1
        timestamp_ns = int(timestamp_sec * 1e9)
        bag.write(topic, serialize_message(navsatfix_msg), timestamp_ns)


def save_gps_vel_data(bag, kitti, gps_frame_id, topic, time_offset=0.0):
    print("Exporting GPS velocity data")
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        twist_msg = TwistStamped()
        twist_msg.header.frame_id = gps_frame_id
        timestamp_sec = float(timestamp.strftime("%s.%f")) + time_offset
        twist_msg.header.stamp = stamp_to_ros_time(timestamp_sec)
        twist_msg.twist.linear.x = oxts.packet.vf
        twist_msg.twist.linear.y = oxts.packet.vl
        twist_msg.twist.linear.z = oxts.packet.vu
        twist_msg.twist.angular.x = oxts.packet.wf
        twist_msg.twist.angular.y = oxts.packet.wl
        twist_msg.twist.angular.z = oxts.packet.wu
        timestamp_ns = int(timestamp_sec * 1e9)
        bag.write(topic, serialize_message(twist_msg), timestamp_ns)


def process_single_drive(bag, args, date, drive, cameras, bridge, time_offset=0.0):
    """Process a single drive and add it to the bag
    
    Args:
        time_offset: Time offset in seconds to add to all timestamps for continuity
    """
    print("\n" + "="*60)
    print("Processing drive: {} - {}".format(date, drive))
    if time_offset > 0:
        print("Time offset: {:.3f} seconds".format(time_offset))
    print("="*60)
    
    try:
        kitti = pykitti.raw(args.dir, date, drive)
    except FileNotFoundError as e:
        print('Error loading dataset: {}'.format(e))
        print('Expected directory structure:')
        print('  {}/'.format(args.dir))
        print('    {}/'.format(date))
        print('      calib_cam_to_cam.txt')
        print('      calib_imu_to_velo.txt')
        print('      calib_velo_to_cam.txt')
        print('      {}_drive_{}_sync/'.format(date, drive))
        return False
    
    if not os.path.exists(kitti.data_path):
        print('Path {} does not exist. Skipping this drive.'.format(kitti.data_path))
        return False

    if len(kitti.timestamps) == 0:
        print('Dataset is empty. Skipping this drive.')
        return False

    # IMU
    imu_frame_id = 'imu_link'
    imu_topic = '/kitti/oxts/imu'
    gps_fix_topic = '/kitti/oxts/gps/fix'
    gps_vel_topic = '/kitti/oxts/gps/vel'
    velo_frame_id = 'velo_link'
    velo_topic = '/kitti/velo'

    T_base_link_to_imu = np.eye(4, 4)
    T_base_link_to_imu[0:3, 3] = [-2.71/2.0-0.05, 0.32, 0.93]

    # tf_static
    transforms = [
        ('base_link', imu_frame_id, T_base_link_to_imu),
        (imu_frame_id, velo_frame_id, inv(kitti.calib.T_velo_imu)),
        (imu_frame_id, cameras[0][1], inv(kitti.calib.T_cam0_imu)),
        (imu_frame_id, cameras[1][1], inv(kitti.calib.T_cam1_imu)),
        (imu_frame_id, cameras[2][1], inv(kitti.calib.T_cam2_imu)),
        (imu_frame_id, cameras[3][1], inv(kitti.calib.T_cam3_imu))
    ]

    util = pykitti.utils.read_calib_file(os.path.join(kitti.calib_path, 'calib_cam_to_cam.txt'))

    # Export
    save_static_transforms(bag, transforms, kitti.timestamps, time_offset)
    save_dynamic_tf(bag, kitti, args.kitti_type, initial_time=None, time_offset=time_offset)
    save_imu_data(bag, kitti, imu_frame_id, imu_topic, time_offset)
    save_gps_fix_data(bag, kitti, imu_frame_id, gps_fix_topic, time_offset)
    save_gps_vel_data(bag, kitti, imu_frame_id, gps_vel_topic, time_offset)
    for camera in cameras:
        save_camera_data(bag, args.kitti_type, kitti, util, bridge, camera=camera[0], camera_frame_id=camera[1], topic=camera[2], initial_time=None, time_offset=time_offset)
    save_velo_data(bag, kitti, velo_frame_id, velo_topic, time_offset)
    
    # Return the last timestamp for this drive (for calculating next offset)
    last_timestamp = float(kitti.timestamps[-1].strftime("%s.%f")) + time_offset
    
    return True, last_timestamp


def run_kitti2bag():
    parser = argparse.ArgumentParser(description="Convert KITTI dataset to ROS2 bag file the easy way!")
    # Accepted argument values
    kitti_types = ["raw_synced", "odom_color", "odom_gray"]
    odometry_sequences = []
    for s in range(22):
        odometry_sequences.append(str(s).zfill(2))
    
    parser.add_argument("kitti_type", choices=kitti_types, help="KITTI dataset type")
    parser.add_argument("dir", nargs="?", default=os.getcwd(), help="base directory of the dataset, if no directory passed the default is current working directory")
    parser.add_argument("-t", "--date", help="date of the raw dataset (i.e. 2011_09_26), option is only for RAW datasets.")
    parser.add_argument("-r", "--drive", help="drive number of the raw dataset (i.e. 0001), option is only for RAW datasets.")
    parser.add_argument("-s", "--sequence", choices=odometry_sequences, help="sequence of the odometry dataset (between 00 - 21), option is only for ODOMETRY datasets.")
    parser.add_argument("-m", "--multi-drive", nargs='+', help="multiple drives to merge into single bag (e.g., -m 0001 0002 0003). Must be used with --date for RAW datasets.")
    parser.add_argument("-o", "--output", help="output bag file name (without extension)")
    parser.add_argument("-f", "--format", choices=['sqlite3', 'mcap'], default='mcap', help="bag storage format (default: mcap)")
    args = parser.parse_args()

    bridge = CvBridge()
    
    # CAMERAS
    cameras = [
        (0, 'camera_gray_left', '/kitti/camera_gray_left'),
        (1, 'camera_gray_right', '/kitti/camera_gray_right'),
        (2, 'camera_color_left', '/kitti/camera_color_left'),
        (3, 'camera_color_right', '/kitti/camera_color_right')
    ]

    if args.kitti_type.find("raw") != -1:
        if args.date == None:
            print("Date option is not given. It is mandatory for raw dataset.")
            print("Usage for raw dataset: kitti2bag raw_synced [dir] -t <date> -r <drive>")
            print("For multiple drives: kitti2bag raw_synced [dir] -t <date> -m <drive1> <drive2> ...")
            sys.exit(1)
        
        # Determine drives to process
        if args.multi_drive:
            drives = args.multi_drive
            if args.output:
                bag_name = args.output
            else:
                bag_name = "kitti_{}_drives_{}_{}".format(args.date, "_".join(drives), args.kitti_type[4:])
        elif args.drive:
            drives = [args.drive]
            if args.output:
                bag_name = args.output
            else:
                bag_name = "kitti_{}_drive_{}_{}".format(args.date, args.drive, args.kitti_type[4:])
        else:
            print("Drive option is not given. Use -r for single drive or -m for multiple drives.")
            print("Usage for raw dataset: kitti2bag raw_synced [dir] -t <date> -r <drive>")
            print("For multiple drives: kitti2bag raw_synced [dir] -t <date> -m <drive1> <drive2> ...")
            sys.exit(1)
        
        # ROS2 bag writer setup with format selection
        writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=bag_name, storage_id=args.format)
        converter_options = rosbag2_py.ConverterOptions('', '')
        writer.open(storage_options, converter_options)
        
        # Create topic metadata for all topics
        topic_metadata_map = {
            '/tf': 'tf2_msgs/msg/TFMessage',
            '/tf_static': 'tf2_msgs/msg/TFMessage',
            '/kitti/oxts/imu': 'sensor_msgs/msg/Imu',
            '/kitti/oxts/gps/fix': 'sensor_msgs/msg/NavSatFix',
            '/kitti/oxts/gps/vel': 'geometry_msgs/msg/TwistStamped',
            '/kitti/velo/pointcloud': 'sensor_msgs/msg/PointCloud2',
        }
        
        for camera in cameras:
            topic_metadata_map[camera[2] + '/image_raw'] = 'sensor_msgs/msg/Image'
            topic_metadata_map[camera[2] + '/camera_info'] = 'sensor_msgs/msg/CameraInfo'
        
        for topic, msg_type in topic_metadata_map.items():
            topic_metadata = rosbag2_py.TopicMetadata(name=topic, type=msg_type, serialization_format='cdr')
            writer.create_topic(topic_metadata)
        
        # Create a wrapper class to match rosbag API
        class BagWriter:
            def __init__(self, writer):
                self.writer = writer
                
            def write(self, topic, serialized_msg, timestamp_ns):
                self.writer.write(topic, serialized_msg, timestamp_ns)
        
        bag = BagWriter(writer)
        
        try:
            success_count = 0
            time_offset = 0.0  # Start with no offset for first drive
            
            for i, drive in enumerate(drives):
                result = process_single_drive(bag, args, args.date, drive, cameras, bridge, time_offset)
                if result:
                    if isinstance(result, tuple):
                        success, last_timestamp = result
                        if success:
                            success_count += 1
                            # For next drive, add a small gap (0.1 seconds) after the last timestamp
                            if i < len(drives) - 1:  # Not the last drive
                                first_timestamp_next = float(
                                    pykitti.raw(args.dir, args.date, drives[i + 1]).timestamps[0].strftime("%s.%f")
                                )
                                # Calculate offset: last_timestamp + gap - first_timestamp_next
                                time_offset = last_timestamp + 0.1 - first_timestamp_next
                    else:
                        if result:
                            success_count += 1
            
            print("\n" + "="*60)
            print("Successfully processed {} out of {} drives".format(success_count, len(drives)))
            print("="*60)

        finally:
            print("\n## OVERVIEW ##")
            print("Bag file created: {}".format(bag_name))
            print("Format: {}".format(args.format))
            del writer
            
    elif args.kitti_type.find("odom") != -1:
        if args.sequence == None:
            print("Sequence option is not given. It is mandatory for odometry dataset.")
            print("Usage for odometry dataset: kitti2bag {odom_color, odom_gray} [dir] -s <sequence>")
            sys.exit(1)
        
        if args.output:
            bag_name = args.output
        else:
            bag_name = "kitti_data_odometry_{}_sequence_{}".format(args.kitti_type[5:], args.sequence)
        
        # ROS2 bag writer setup with format selection
        writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=bag_name, storage_id=args.format)
        converter_options = rosbag2_py.ConverterOptions('', '')
        writer.open(storage_options, converter_options)
        
        # Determine which cameras to use
        if args.kitti_type.find("gray") != -1:
            used_cameras = cameras[:2]
        elif args.kitti_type.find("color") != -1:
            used_cameras = cameras[-2:]
        
        # Create topic metadata
        topic_metadata_map = {
            '/tf': 'tf2_msgs/msg/TFMessage',
        }
        
        for camera in used_cameras:
            topic_metadata_map[camera[2] + '/image_rect'] = 'sensor_msgs/msg/Image'
            topic_metadata_map[camera[2] + '/camera_info'] = 'sensor_msgs/msg/CameraInfo'
        
        for topic, msg_type in topic_metadata_map.items():
            topic_metadata = rosbag2_py.TopicMetadata(name=topic, type=msg_type, serialization_format='cdr')
            writer.create_topic(topic_metadata)
        
        class BagWriter:
            def __init__(self, writer):
                self.writer = writer
                
            def write(self, topic, serialized_msg, timestamp_ns):
                self.writer.write(topic, serialized_msg, timestamp_ns)
        
        bag = BagWriter(writer)
        
        kitti = pykitti.odometry(args.dir, args.sequence)
        if not os.path.exists(kitti.sequence_path):
            print('Path {} does not exist. Exiting.'.format(kitti.sequence_path))
            sys.exit(1)

        kitti.load_calib()
        kitti.load_timestamps()
             
        if len(kitti.timestamps) == 0:
            print('Dataset is empty? Exiting.')
            sys.exit(1)
            
        if args.sequence in odometry_sequences[:11]:
            print("Odometry dataset sequence {} has ground truth information (poses).".format(args.sequence))
            kitti.load_poses()

        try:
            util = pykitti.utils.read_calib_file(os.path.join(args.dir, 'sequences', args.sequence, 'calib.txt'))
            current_epoch = (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()
            
            # Export
            save_dynamic_tf(bag, kitti, args.kitti_type, initial_time=current_epoch)
            for camera in used_cameras:
                save_camera_data(bag, args.kitti_type, kitti, util, bridge, camera=camera[0], camera_frame_id=camera[1], topic=camera[2], initial_time=current_epoch)

        finally:
            print("## OVERVIEW ##")
            print("Bag file created: {}".format(bag_name))
            print("Format: {}".format(args.format))
            del writer


if __name__ == '__main__':
    run_kitti2bag()