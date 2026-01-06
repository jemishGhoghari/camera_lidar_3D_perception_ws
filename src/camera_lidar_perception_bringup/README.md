# camera_lidar_perception_bringup

Launch and configuration package for bringing up a camera–LiDAR perception pipeline and a Nav2 map server in ROS 2.

## Overview

This package provides:

- A bringup launch that starts a multi-threaded rclcpp component container for perception.
- Foxglove layout configuration in JSON format to import into foxglove studio app.

Key dependencies (declared in package.xml):

- ament_cmake
- camera_lidar_3d_perception
- yolo11_inference

## Launch files

- launch/camera_lidar_perception_bringup.launch.py

  - Starts rclcpp_components: component_container_mt.
  - Optional visualization via Foxglove.

## Launch arguments

camera_lidar_perception_bringup.launch.py:

- visualize: true|false to start Foxglove bridge with the provided config.


## RViz

RViz config:

- rviz/rviz_config.rviz

## Foxglove Studio

- foxglove/foxglove_layout.json

Enable with visualize:=true on the bringup launch.

## Examples Usage

- ros2 launch camera_lidar_perception_bringup camera_lidar_perception_bringup.launch.py visualize:=true
