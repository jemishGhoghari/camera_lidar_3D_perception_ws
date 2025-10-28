# camera_lidar_perception_bringup

Launch and configuration package for bringing up a camera–LiDAR perception pipeline and a Nav2 map server in ROS 2.

## Overview

This package provides:

- A bringup launch that starts a multi-threaded rclcpp component container for perception.
- A Nav2 map server launch for static map loading.
- Ready-to-use maps and an RViz configuration.

Key dependencies (declared in package.xml):

- ament_cmake
- camera_lidar_3d_perception
- nav2_map_server
- yolo11_inference

## Launch files

- launch/camera_lidar_perception_bringup.launch.py

  - Starts rclcpp_components: component_container_mt.
  - Optional visualization via RViz.
- launch/nav2_map_server.launch.py

  - Starts nav2_map_server: map_server.
  - Supports auto activation and custom map YAML.

## Launch arguments

camera_lidar_perception_bringup.launch.py:

- map_environment: Selects a bundled map environment (e.g., bathroom, office).
- visualize: true|false to start RViz with the provided config.

nav2_map_server.launch.py:

- yaml_filename: Path to a .yaml map (overrides map_environment).
- auto_activate: true|false to auto-activate the map server lifecycle node.

Note: Available map_environment values match the directories under maps/ (bathroom, office).

## Maps

Bundled maps:

- maps/bathroom/bathroom.yaml (+ .pgm)
- maps/office/office.yaml (+ .pgm)

## RViz

RViz config:

- rviz/rviz_config.rviz

Enable with visualize:=true on the bringup launch.

## Examples

- ros2 launch camera_lidar_perception_bringup camera_lidar_perception_bringup.launch.py map_environment:=bathroom visualize:=true

## Occupancy Grid Generator
The Occupancy Grid Generator generates the 3D bounding box overlay on the raw images. It uses the nav2_map_server to load the static map and the camera_lidar_3d_perception package to create overlay.
It also writes the JSON file with all detected objects with pose, hypothesis, class_id and timestamp.

The example JSON format look like below:
```bash
{
  "frame_id": "map",
  "detections": [
    {
      "id": "obj_id",
      "class": "class_id",
      "pose": {
        "x": x_pos,
        "y": y_pos,
        "yaw": z_pos
      },
      "dimensions": {
        "length": length,
        "width": width
      },
      "confidence": confidence,
      "count": count
    }
  ]
}
```

### Usage
```bash
ros2 run camera_lidar_perception_bringup ocp_generator.py
```

In order to generate the 

Note: The default parameters are set to comply with the current setup, so you do not need to override parameters.

