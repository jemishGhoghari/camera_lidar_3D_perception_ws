# YOLO11_inference

`yolo11_inference` is a real-time object detection node for ROS 2 based on  YOLO11, running with the OpenCV DNN backend for efficient deployment on CPU or CUDA-enabled GPU. The node performs 2D object detection by subscribing to a compressed RGB image topic and publishing `vision_msgs/Detection2DArray` messages that contain:

* Bounding boxes in pixel coordinates
* Detected class IDs and labels
* Confidence scores for each detection

This node is designed as a ComposableNode so it can run standalone or be integrated into a larger perception pipeline. It supports configurable model input resolution, confidence filtering, and class filtering at runtime without recompilation. The node expects a  YOLO11 ONNX model , which can be exported using Ultralytics tools.

---

## Overview

- **Input**: `sensor_msgs/CompressedImage` (RGB)
- **Output**: `vision_msgs/Detection2DArray`
- **Backend**: OpenCV DNN with optional CUDA acceleration
- **Model**: YOLO11 ONNX (example included under `weights/`)

The repository also includes a small **overlay visualizer** script to draw detections on images in real time.

---

## Package structure

```
yolo11_inference/
├── config/
│   └── params.yaml
├── include/
│   └── yolo11_inference/
│       ├── yolo11_inference_node.hpp
│       └── yolo_dnn_inference.hpp
├── launch/
│   └── yolo11_inference_node.launch.py
├── script/
│   └── viz.py
├── src/
│   ├── yolo11_inference_node.cpp
│   └── yolo_dnn_inference.cpp
├── weights/
│   └── yolo11m_1280x736.onnx
├── CMakeLists.txt
├── LICENSE
├── package.xml
└── README.md
```

**Key files**

- `src/yolo11_inference_node.cpp` — ROS 2 component wrapping the detector
- `src/yolo_dnn_inference.cpp` — OpenCV DNN inference, post‑processing, NMS, letterbox
- `include/yolo11_inference/*.hpp` — public headers
- `launch/yolo11_inference_node.launch.py` — composable node container launch
- `config/params.yaml` — example class list filter
- `script/viz.py` — Detection2D overlay visualizer
- `weights/yolo11m_1280x736.onnx` — sample ONNX model

---

## Dependencies

Declared in `CMakeLists.txt` and `package.xml`:

- `rclcpp_components`
- `sensor_msgs`
- `vision_msgs`
- `image_transport`
- `cv_bridge`
- **OpenCV** (with DNN; CUDA optional if OpenCV is built with CUDA)
- `ament_cmake_auto`

> Ensure `vision_msgs` is installed and matches your ROS 2 distribution.

---

## Build

```bash
# In your ROS 2 workspace
colcon build --packages-select yolo11_inference --symlink-install
source install/setup.bash
```

---

## Parameters

The node declares the following parameters (defaults inferred from source):

| Name                        | Type     |                          Default | Description                             |
| --------------------------- | -------- | -------------------------------: | --------------------------------------- |
| `network_image_width`     | int      |                              640 | Network input width after letterboxing  |
| `network_image_height`    | int      |                              640 | Network input height after letterboxing |
| `model_file_path`         | string   |                           `""` | Path to YOLO11 ONNX model               |
| `confidence_threshold`    | double   |                             0.25 | Score threshold before NMS              |
| `nms_threshold`           | double   |                             0.45 | IOU threshold for NMS                   |
| `use_cuda`                | bool     |                             true | Use CUDA target if available in OpenCV  |
| `class_names`             | string[] |      `[]` (or via params file) | Optional whitelist of labels to keep    |
| `detections_output_topic` | string   | `/yolo11_inference/detections` | Output topic for `Detection2DArray`   |

**Note**
The current implementation subscribes to **`sensor_msgs/CompressedImage`**. When using `image_transport:=compressed`, set the input topic to the `.../compressed` subtopic. If you prefer raw images (`sensor_msgs/Image`), the subscriber code must be adapted.

Example class filter (`config/params.yaml`):

```yaml
/**:
  ros__parameters:
    class_names: ["couch", "dining table", "chair", "potted plant", "bench"]
```

---

## Topics

- **Subscribed**: **image_input_topic** (launch arg) → expects `sensor_msgs/CompressedImage`.Example: `/zed/zed_node/rgb/image_rect_color/compressed`
- **Published**: **detections_output_topic** (launch arg) → `vision_msgs/Detection2DArray`.
  Each `Detection2D` contains a bounding box in **pixel coordinates** and `ObjectHypothesisWithPose` entries with class ID and score.

---

## How to run

### 1) Using the provided launch file (recommended)

```bash
ros2 launch yolo11_inference yolo11_inference_node.launch.py \
  component_container_name:=yolov11_inference_container \
  standalone:=true \
  image_input_topic:=/zed/zed_node/rgb/image_rect_color/compressed \
  detections_output_topic:=/yolo11_inference/detections \
  model_file_path:=$(ros2 pkg prefix yolo11_inference)/share/yolo11_inference/weights/yolo11m_1280x736.onnx \
  network_image_width:=1280 \
  network_image_height:=736 \
  confidence_threshold:=0.45 \
  nms_threshold:=0.50 \
  use_cuda:=True
```

**Note**: You check all available parameters of the launch file by executing following command:

`ros2 launch yolo11_inference yolo11_inference_node.launch.py -s`

### 2) Running the overlay visualizer

```bash
ros2 run yolo11_inference viz.py --ros-args \
  -p image_topic:=/zed/zed_node/rgb/image_rect_color/compressed \
  -p detection_topic:=/yolo11_inference/detections \
  -p show_window:=true
```

This subscribes to the compressed image and overlays the latest `Detection2DArray` on it. It also republishes to `detection2d_overlay` for viewing in RViz/ImageView.

---

## Notes on models and export

- Use **Ultralytics YOLO11** and export to ONNX with opset ≥ 12.
- OpenCV DNN supports a broad subset of ONNX ops. If you hit a `Concat` shape error, verify your export input size and that all branches match shapes.

Example export (tune to your model):

```bash
yolo export model=yolo11m.pt format=onnx imgsz=736,1280 opset=12 dynamic=False simplify=False
```

Match `network_image_width` and `network_image_height` to these sizes.

---

## Example integration (ZED)

```bash
# Ensure ZED publishes compressed images (if your camera outputs raw)
ros2 run image_transport republish raw \
  in:=/zed/zed_node/rgb/image_rect_color \
  compressed out:=/zed/zed_node/rgb/image_rect_color

# Launch detector
ros2 launch yolo11_inference yolo11_inference_node.launch.py \
  image_input_topic:=/zed/zed_node/rgb/image_rect_color/compressed \
  model_file_path:=$(ros2 pkg prefix yolo11_inference)/share/yolo11_inference/weights/yolo11m_1280x736.onnx
```

---

## Code style & build

This package uses `ament_cmake_auto` and installs the launch, config and weights directories for runtime access. The visualizer script is installed to `lib/yolo11_inference` and can be run via `ros2 run` as shown above.

---
