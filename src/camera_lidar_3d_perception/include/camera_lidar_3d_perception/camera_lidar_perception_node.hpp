#ifndef CAMERA_LIDAR_PERCEPTION_NODE_HPP
#define CAMERA_LIDAR_PERCEPTION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include "camera_lidar_3d_perception/camera_lidar_perception_node.hpp"

#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Dense>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::PointCloud2, vision_msgs::msg::Detection2DArray, sensor_msgs::msg::CameraInfo>
    ApproximateSyncPolicy;

/**
 * @file camera_lidar_perception_node.hpp
 * @brief Header file for the CameraLidarPerceptionNode class, which integrates camera and LiDAR data for 3D perception.
 */

namespace camera_lidar_3d_perception
{

/**
 * @class CameraLidarPerceptionNode
 * @brief A ROS 2 node for synchronizing and processing camera and LiDAR data to generate 3D detections.
 */
class CameraLidarPerceptionNode : public rclcpp::Node
{
public:
  /**
   * @brief Constructor for the CameraLidarPerceptionNode class.
   * @param options Node options for configuring the ROS 2 node.
   */
  explicit CameraLidarPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  /**
   * @brief Destructor for the CameraLidarPerceptionNode class.
   */
  ~CameraLidarPerceptionNode();

  /**
   * @brief Callback function to synchronize and process PointCloud2, Detection2DArray, and CameraInfo messages.
   * @param pointcloud_msg Pointer to the incoming PointCloud2 message.
   * @param detections_msg Pointer to the incoming Detection2DArray message.
   * @param camera_info_msg Pointer to the incoming CameraInfo message.
   */
  void syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_msg,
                    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections_msg,
                    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg);

  /**
   * @brief Transforms a point cloud from the LiDAR frame to the camera frame.
   * @param cloud_in Input point cloud in the LiDAR frame.
   * @param cloud_out Output point cloud in the camera frame.
   * @param transform Transformation matrix from LiDAR to camera frame.
   */
  void transformPointCloudToCameraFrame(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                        const Eigen::Affine3f& transform);

  /**
   * @brief Projects a point cloud onto the camera image plane and associates it with 2D detections.
   * @param point_cloud Input point cloud.
   * @param yolo_detections_msg YOLO 2D detections message.
   * @param header Header for the output messages.
   * @param detection3d_array_msg Output 3D detections message.
   * @param combine_detection_cloud_msg Output combined detection point cloud message.
   */
  void projectCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
                    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& yolo_detections_msg,
                    const std_msgs::msg::Header& header, vision_msgs::msg::Detection3DArray& detection3d_array_msg,
                    sensor_msgs::msg::PointCloud2& combine_detection_cloud_msg);

  /**
   * @brief Extracts points from a point cloud that fall within a 2D bounding box.
   * @param cloud Input point cloud.
   * @param detection2d_msg 2D detection message containing the bounding box.
   * @param raw_detection_cloud Output point cloud containing points within the bounding box.
   */
  void processPointsWithBbox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                             const vision_msgs::msg::Detection2D& detection2d_msg,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_detection_cloud);

  /**
   * @brief Creates 3D bounding boxes from a point cloud and associated detection results.
   * @param detection3d_array_msg Output 3D detections message.
   * @param cloud Input point cloud.
   * @param detections_results_msg Vector of object hypotheses with poses.
   */
  void createBoundingBox(vision_msgs::msg::Detection3DArray& detection3d_array_msg,
                         const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                         const std::vector<vision_msgs::msg::ObjectHypothesisWithPose>& detections_results_msg);

  /**
   * @brief Downsamples a PointCloud2 message to reduce its size.
   * @param cloud_msg Input PointCloud2 message.
   * @return Downsampled point cloud.
   */
  pcl::PointCloud<pcl::PointXYZ>::Ptr
  downSampleCloudMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg);

  /**
   * @brief Transforms a point cloud from one frame to another.
   * @param cloud_msg Input point cloud.
   * @param source_frame Source frame of the point cloud.
   * @param target_frame Target frame to transform the point cloud into.
   * @param stamp Timestamp of the transformation.
   * @return Transformed point cloud.
   */
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2TransformedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_msg,
                                                             const std::string& source_frame,
                                                             const std::string& target_frame,
                                                             const rclcpp::Time& stamp);

  /**
   * @brief Performs Euclidean clustering on a point cloud to segment it into clusters.
   * @param cloud Input point cloud.
   * @return Point cloud containing clustered points.
   */
  pcl::PointCloud<pcl::PointXYZ>::Ptr eucludianClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

  /**
   * @brief Creates a MarkerArray message for visualizing 3D detections in RViz OR Optionally Foxglove Studio.
   * @param detection3d_array_msg Input 3D detections message.
   * @param duration Duration for which the markers should be displayed.
   * @return MarkerArray message for visualization.
   */
  visualization_msgs::msg::MarkerArray
  createMarkerArray(const vision_msgs::msg::Detection3DArray& detection3d_array_msg, const double& duration);

private:
  /**
   * @brief Subscriber for PointCloud2 messages.
   */
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pointcloud_sub_;

  /**
   * @brief Subscriber for Detection2DArray messages.
   */
  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> detection_2d_sub_;

  /**
   * @brief Subscriber for CameraInfo messages.
   */
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  /**
   * @brief Synchronizer for synchronizing PointCloud2, Detection2DArray, and CameraInfo messages.
   */
  std::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>> sync_;

  /**
   * @brief Publisher for Detection3DArray messages.
   */
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_3d_pub_;
};

}  // namespace camera_lidar_3d_perception

#endif  // CAMERA_LIDAR_PERCEPTION_NODE_HPP