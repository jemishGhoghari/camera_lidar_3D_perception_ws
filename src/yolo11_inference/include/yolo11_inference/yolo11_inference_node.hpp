#ifndef YOLO11_INFERENCE_NODE
#define YOLO11_INFERENCE_NODE

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/image_transport.hpp>

#include <cv_bridge/cv_bridge.h>

#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include <sensor_msgs/msg/image.hpp>

#include "yolo_dnn_inference.hpp"

namespace yolo11_inference
{

class Yolo11InferenceNode : public rclcpp::Node
{
public:
  explicit Yolo11InferenceNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~Yolo11InferenceNode();

private:
  void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg);

  std::unique_ptr<Yolo11DNNInference> inference_engine_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;

private:
  YoloDNNConfig yolo_config_;
  std::vector<std::string> class_names_;
  std::vector<std::string> class_names_coco_;
};

}  // namespace yolo11_inference

#endif  // YOLO11_INFERENCE_NODE