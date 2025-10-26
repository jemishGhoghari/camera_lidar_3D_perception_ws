#include <yolo11_inference/yolo11_inference_node.hpp>

namespace yolo11_inference
{

Yolo11InferenceNode::Yolo11InferenceNode(const rclcpp::NodeOptions& options) : Node("yolo11_inference_node", options)
{
  yolo_config_.networkInputWidth = this->declare_parameter<int>("network_image_width", 640);
  yolo_config_.networkInputHeight = this->declare_parameter<int>("network_image_height", 640);
  yolo_config_.modelPath = this->declare_parameter<std::string>("model_file_path", "");
  yolo_config_.confThreshold = this->declare_parameter<float>("confidence_threshold", 0.25f);
  yolo_config_.nmsThreshold = this->declare_parameter<float>("nms_threshold", 0.45f);
  yolo_config_.use_cuda = this->declare_parameter<bool>("use_cuda", true);

  class_names_ = this->declare_parameter<std::vector<std::string>>("class_names", std::vector<std::string>{});
  class_names_coco_ = loadCOCOClasses();

  // Initialize YOLO DNN Inference Engine
  inference_engine_ = std::make_unique<Yolo11DNNInference>(yolo_config_, class_names_, this->get_logger());

  // Publishers
  detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);

  // Subscribers
  image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "image_raw", rclcpp::SensorDataQoS(),
      std::bind(&Yolo11InferenceNode::imageCallback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "Yolo11 Inference Node has been initialized");
}

Yolo11InferenceNode::~Yolo11InferenceNode()
{
}

void Yolo11InferenceNode::imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg)
{
  // Decode JPEG/PNG
  cv::Mat buf(1, static_cast<int>(msg->data.size()), CV_8UC1, const_cast<uint8_t*>(msg->data.data()));
  cv::Mat frame = cv::imdecode(buf, cv::IMREAD_COLOR);
  if (frame.empty())
  {
    RCLCPP_WARN(get_logger(), "Failed to decode CompressedImage");
    return;
  }

  DetectionResult dets;
  if (!inference_engine_->infer(frame, dets))
  {
    return;
  }

  auto det_array = vision_msgs::msg::Detection2DArray();
  det_array.header = msg->header;
  det_array.detections.reserve(dets.boxes.size());

  for (size_t i = 0; i < dets.boxes.size(); ++i)
  {
    const cv::Rect& b = dets.boxes[i];
    const float confidence_score = dets.confidence_scores[i];
    const int class_id = dets.class_ids[i];

    vision_msgs::msg::Detection2D det;
    det.header = msg->header;
    det.bbox.center.position.x = b.x + b.width * 0.5f;
    det.bbox.center.position.y = b.y + b.height * 0.5f;
    det.bbox.size_x = b.width;
    det.bbox.size_y = b.height;

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    if (class_id >= 0 && class_id < (int)class_names_coco_.size())
    {
      hyp.hypothesis.class_id = class_names_coco_[class_id];
    }
    else
    {
      hyp.hypothesis.class_id = "cls_" + std::to_string(class_id);
    }

    hyp.hypothesis.score = confidence_score;
    det.results.push_back(hyp);
    det_array.detections.push_back(det);
  }

  detection_pub_->publish(std::move(det_array));
}

}  // namespace yolo11_inference

// Register node as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(yolo11_inference::Yolo11InferenceNode)