# camera_lidar_3d_perception

A ROS 2 package for 3D perception by combining camera detections with LiDAR point clouds. It associates 2D bounding boxes from a camera with corresponding 3D LiDAR data to extract object point clusters in real time.

---

## Features

- Synchronizes LiDAR point cloud and camera detections
- Projects LiDAR points into camera frame
- Filters point cloud using 2D bounding box
- VoxelGrid downsampling (PCL)
- Euclidean clustering for object extraction
- Publishes filtered 3D detections

---

## Package Structure

```
├── include/camera_lidar_3d_perception
│ └── camera_lidar_perception_node.hpp
├── src
│ └── camera_lidar_perception_node.cpp
├── launch
│ └── camera_lidar_perception_node.launch.py
├── config
│ └── config.yaml
├── package.xml
├── CMakeLists.txt
└── LICENSE
```

---

## Topics

**Subscriptions**

- `pointcloud` → `sensor_msgs/PointCloud2`
- `detections` → `vision_msgs/Detection2DArray`
- `camera_info` → `sensor_msgs/CameraInfo`

**Publications**

- `detections_3d` → `vision_msgs/Detection3DArray`

> All names above are the node’s internal names. Use launch arguments to remap to your actual topics.

---

## Parameters (set via launch file)

- `voxel_leaf_size` (double, default `0.1`) — Voxel grid size for downsampling.
- `cluster_tolerance` (double, default `0.1`) — Euclidean clustering tolerance.
- `min_cluster_size` (int, default `10`) — Minimum cluster size.
- `max_cluster_size` (int, default `25000`) — Maximum cluster size.

---

## Build

```bash
# In your ROS 2 workspace
colcon build --packages-select camera_lidar_3d_perception --symlink-install
source install/setup.bash

```

---
# Launch & Usage

## Launch Arguments
| Name                       | Default                           | Description                                   |
|---------------------------|-----------------------------------|-----------------------------------------------|
| component_container_name  | camera_lidar_perception_container | Target container name                          |
| standalone                | True                              | Create a container and load the node if True  |
| pointcloud_topic          | pointcloud                        | LiDAR PointCloud2 input                        |
| detection_2d_input_topic  | detections                        | 2D detections input                            |
| detections_3d_output_topic| detections_3d                     | 3D detections output                           |
| camera_info_topic         | camera_info                        | Camera intrinsics input                         |
| voxel_leaf_size           | 0.1                               | Voxel grid size                                |
| cluster_tolerance         | 0.1                               | Clustering tolerance                           |
| min_cluster_size          | 10                                | Minimum cluster size                            |
| max_cluster_size          | 25000                             | Maximum cluster size                            |

## Default(Standalone container):

start a multithreaded component container and load the node:
```bash
ros2 launch camera_lidar_3d_perception camera_lidar_perception_node.launch.py
```
## Common overrides:
```bash
# Parameter overrides
ros2 launch camera_lidar_3d_perception camera_lidar_perception_node.launch.py \
    voxel_leaf_size:=0.1 \
    cluster_tolerance:=0.1 \
    min_cluster_size:=10 \
    max_cluster_size:=25000 \
```
---
```bash
# Topic remapping
ros2 launch camera_lidar_3d_perception camera_lidar_perception_node.launch.py \
  pointcloud_topic:=/livox/lidar \
  detection_2d_input_topic:=/yolo/detections \
  camera_info_topic:=/zed/zed_node/rgb/camera_info \
  detections_3d_output_topic:=/perception/detections_3d
```
---
## Load into existing component container:
```bash
ros2 launch camera_lidar_3d_perception camera_lidar_perception_node.launch.py \
  component_container_name:=<your_container_name> \
  standalone:=false
```
---

# Dependencies
- rclcpp
- rclcpp_components
- sensor_msgs
- vision_msgs
- message_filters
- PCL
- pcl_ros
- pcl_conversions
- tf2
- tf2_ros
