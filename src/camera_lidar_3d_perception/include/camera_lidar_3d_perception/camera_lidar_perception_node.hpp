#ifndef CAMERA_LIDAR_PERCEPTION_NODE_HPP
#define CAMERA_LIDAR_PERCEPTION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include "camera_lidar_3d_perception/camera_lidar_perception_node.hpp"

#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::PointCloud2, vision_msgs::msg::Detection2DArray, sensor_msgs::msg::CameraInfo>
    ApproximateSyncPolicy;

namespace camera_lidar_3d_perception
{

class CameraLidarPerceptionNode : public rclcpp::Node
{
public:
  explicit CameraLidarPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~CameraLidarPerceptionNode();

  // void detection2DCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr msg);
  void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_msg,
                    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections_msg,
                    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg);

private:
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pointcloud_sub_;
  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> detection_2d_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  std::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>> sync_;

  // rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_2d_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_3d_pub_;
};

}  // namespace camera_lidar_3d_perception

#endif  // CAMERA_LIDAR_PERCEPTION_NODE_HPP