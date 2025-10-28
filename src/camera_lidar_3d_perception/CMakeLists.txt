cmake_minimum_required(VERSION 3.8)
project(camera_lidar_3d_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(cv_bridge REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(camera_lidar_perception_node SHARED 
  src/camera_lidar_perception_node.cpp
)
ament_target_dependencies(camera_lidar_perception_node 
  "rclcpp_components" 
  "vision_msgs" 
  "message_filters"
  "pcl_conversions" 
  "pcl_ros" 
  "tf2_eigen" 
  "tf2_ros" 
  "OpenCV" 
  "PCL" 
  "visualization_msgs" 
  "cv_bridge" 
  "geometry_msgs" 
  "image_geometry"
)

rclcpp_components_register_nodes(camera_lidar_perception_node "camera_lidar_3d_perception::CameraLidarPerceptionNode")
set(node_plugins "${node_plugins}camera_lidar_3d_perception::CameraLidarPerceptionNode;$<TARGET_FILE:camera_lidar_3d_perception>\n")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE launch)