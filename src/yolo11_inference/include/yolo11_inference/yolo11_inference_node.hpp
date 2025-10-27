#ifndef YOLO11_INFERENCE_NODE
#define YOLO11_INFERENCE_NODE

// ROS 2 node that performs YOLOv11 DNN inference on incoming compressed images
// and publishes 2D detection results.

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
/**
 * @class Yolo11InferenceNode
 * @brief ROS 2 node wrapping a YOLOv11 DNN inference engine.
 *
 * Subscribes to sensor_msgs/CompressedImage, runs inference, and publishes
 * vision_msgs/Detection2DArray detections.
 */
class Yolo11InferenceNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct the node and set up subscriptions/publishers/inference.
   * @param options Node options for composition and parameter handling.
   */
  explicit Yolo11InferenceNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  /**
   * @brief Clean up resources and shut down the inference engine.
   */
  ~Yolo11InferenceNode();

private:
  /**
 * @brief 
   * @param msg Incoming compressed image message.
   */
  void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg);

  /// YOLOv11 DNN inference engine instance.
  std::unique_ptr<Yolo11DNNInference> inference_engine_;

  /// Subscriber for compressed image input.
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;

  /// Publisher for 2D detection results.
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;

private:
  /// Configuration parameters for the YOLO DNN.
  YoloDNNConfig yolo_config_;

  /// Class names used by the active model.
  std::vector<std::string> class_names_;

  /// COCO class names (if using a COCO-pretrained model).
  std::vector<std::string> class_names_coco_;
};

}  // namespace yolo11_inference

#endif  // YOLO11_INFERENCE_NODE