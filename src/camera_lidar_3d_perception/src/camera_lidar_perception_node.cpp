#include "camera_lidar_3d_perception/camera_lidar_perception_node.hpp"

namespace camera_lidar_3d_perception
{

CameraLidarPerceptionNode::CameraLidarPerceptionNode(const rclcpp::NodeOptions& options)
  : Node("camera_lidar_3d_node", options)
{
  // Subscriber
  pointcloud_sub_.subscribe(this, "pointcloud");
  detection_2d_sub_.subscribe(this, "detections");
  camera_info_sub_.subscribe(this, "camera_info");

  // Initialize Sync policy
  sync_ = std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(20);
  sync_->connectInput(pointcloud_sub_, detection_2d_sub_, camera_info_sub_);
  sync_->registerCallback(std::bind(&CameraLidarPerceptionNode::syncCallback, this, std::placeholders::_1,
                                    std::placeholders::_2, std::placeholders::_3));

  // Publishers
  detection_3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("detections_3d", 10);

  // detection_2d_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
  //     "detections", rclcpp::SensorDataQoS(),
  //     std::bind(&CameraLidarPerceptionNode::detection2DCallback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "Camera LiDAR Node has been initialized");
}

CameraLidarPerceptionNode::~CameraLidarPerceptionNode()
{
}

void CameraLidarPerceptionNode::syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_msg,
                                             const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections_msg,
                                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg)
{
  (void)pointcloud_msg;
  (void)detections_msg;
  (void)camera_info_msg;

  RCLCPP_INFO(get_logger(), "Messages are received");
}

// void CameraLidarPerceptionNode::detection2DCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& msg)
// {
//   (void)msg;
// }

}  // namespace camera_lidar_3d_perception

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(camera_lidar_3d_perception::CameraLidarPerceptionNode)