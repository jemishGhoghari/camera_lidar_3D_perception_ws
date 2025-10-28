cmake_minimum_required(VERSION 3.8)
project(yolo11_inference)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(yolo11_inference_node SHARED 
  src/yolo11_inference_node.cpp 
  src/yolo_dnn_inference.cpp
)
ament_target_dependencies(yolo11_inference_node 
  "rclcpp_components" 
  "OpenCV"
  "sensor_msgs"
  "vision_msgs"
  "image_transport"
  "cv_bridge"
)
rclcpp_components_register_nodes(yolo11_inference_node "yolo11_inference::Yolo11InferenceNode")
set(node_plugins "${node_plugins}yolo11_inference::Yolo11InferenceNode;$<TARGET_FILE:yolo11_inference>\n")

install(
  PROGRAMS
    script/viz.py
  DESTINATION lib/yolo11_inference
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE config launch weights)
