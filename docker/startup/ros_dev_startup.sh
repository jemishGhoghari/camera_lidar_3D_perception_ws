#!/bin/bash

# run isaac_ros docker without building
$HOME/workspaces/camera_lidar_3D_perception_ws/src/isaac_ros_common/scripts/run_dev.sh \
--isaac_ros_dev_dir $HOME/workspaces/camera_lidar_3D_perception_ws \
$@
