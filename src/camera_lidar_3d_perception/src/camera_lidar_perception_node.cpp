#include "camera_lidar_3d_perception/camera_lidar_perception_node.hpp"

namespace camera_lidar_3d_perception
{

CameraLidarPerceptionNode::CameraLidarPerceptionNode(const rclcpp::NodeOptions& options)
  : Node("camera_lidar_3d_node", options)
{
  // Parameter declarations
  voxel_leaf_size_ = this->declare_parameter<double>("voxel_leaf_size", 0.5);
  cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.5);
  min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 100);
  max_cluster_size_ = this->declare_parameter<int>("max_cluster_size", 25000);

  // Subscriber
  pointcloud_sub_.subscribe(this, "pointcloud");
  detection_2d_sub_.subscribe(this, "detections");
  camera_info_sub_.subscribe(this, "camera_info");

  // Initialize Sync policy
  sync_ = std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(10);
  sync_->connectInput(pointcloud_sub_, detection_2d_sub_, camera_info_sub_);
  sync_->registerCallback(std::bind(&CameraLidarPerceptionNode::syncCallback, this, std::placeholders::_1,
                                    std::placeholders::_2, std::placeholders::_3));

  // Publishers
  detection_3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("detections_3d", 10);

  last_callback_time_ = this->now();
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
}

CameraLidarPerceptionNode::~CameraLidarPerceptionNode()
{
}

void CameraLidarPerceptionNode::syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_msg,
                                             const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections_msg,
                                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg)
{
  rclcpp::Time current_time = this->now();
  rclcpp::Duration callback_interval = current_time - last_callback_time_;
  last_callback_time_ = current_time;

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  downsampled_cloud = downSampleCloudMsg(pointcloud_msg);

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  const std::string camera_frame = camera_info_msg->header.frame_id;
  transformed_cloud = cloud2TransformedCloud(downsampled_cloud, pointcloud_msg->header.frame_id, camera_frame,
                                             pointcloud_msg->header.stamp);

  vision_msgs::msg::Detection3DArray detection3d_array_msg;
  sensor_msgs::msg::PointCloud2 detection_cloud_msg;
  projectCloud(transformed_cloud, detections_msg, camera_info_msg, pointcloud_msg->header, detection3d_array_msg,
               detection_cloud_msg);

  detection_3d_pub_->publish(detection3d_array_msg);
}

void CameraLidarPerceptionNode::transformPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                                    const Eigen::Affine3f& transform)
{
  int cloud_size = cloud_in->size();
  cloud_out->resize(cloud_size);

  for (int i = 0; i < cloud_size; i++)
  {
    const auto& point = cloud_in->points[i];
    cloud_out->points[i].x =
        transform(0, 0) * point.x + transform(0, 1) * point.y + transform(0, 2) * point.z + transform(0, 3);
    cloud_out->points[i].y =
        transform(1, 0) * point.x + transform(1, 1) * point.y + transform(1, 2) * point.z + transform(1, 3);
    cloud_out->points[i].z =
        transform(2, 0) * point.x + transform(2, 1) * point.y + transform(2, 2) * point.z + transform(2, 3);
  }
}

void CameraLidarPerceptionNode::projectCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& yolo_detections_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg, const std_msgs::msg::Header& header,
    vision_msgs::msg::Detection3DArray& detection3d_array_msg,
    sensor_msgs::msg::PointCloud2& combine_detection_cloud_msg)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr combine_detection_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  detection3d_array_msg.header = header;
  detection3d_array_msg.header.stamp = yolo_detections_msg->header.stamp;

  for (size_t i = 0; i < yolo_detections_msg->detections.size(); ++i)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
    processPointsWithBbox(point_cloud, yolo_detections_msg->detections[i], camera_info_msg, detection_cloud_raw);

    if (!detection_cloud_raw->points.empty())
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      detection_cloud =
          cloud2TransformedCloud(detection_cloud_raw, camera_info_msg->header.frame_id, "base_link", header.stamp);

      pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_detection_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      clustered_detection_cloud = eucludianClustering(detection_cloud);
      *combine_detection_cloud += *clustered_detection_cloud;

      createBoundingBox(detection3d_array_msg, clustered_detection_cloud, yolo_detections_msg->detections[i].results);
    }
  }
}

bool CameraLidarPerceptionNode::project3DToPixelRectified(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cam_info_msg, const cv::Point3d& points_3d, cv::Point2d& u_v)
{
  const double fx = cam_info_msg->k[0];
  const double fy = cam_info_msg->k[4];
  const double cx = cam_info_msg->k[2];
  const double cy = cam_info_msg->k[5];

  const double invZ = 1.0 / points_3d.z;
  u_v.x = fx * (points_3d.x * invZ) + cx;
  u_v.y = fy * (points_3d.y * invZ) + cy;
  return true;
}

void CameraLidarPerceptionNode::processPointsWithBbox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                      const vision_msgs::msg::Detection2D& detection2d_msg,
                                                      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info,
                                                      pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_detection_cloud)
{
  // Precompute bbox edges in pixels
  const double u_min = detection2d_msg.bbox.center.position.x - detection2d_msg.bbox.size_x * 0.5;
  const double u_max = detection2d_msg.bbox.center.position.x + detection2d_msg.bbox.size_x * 0.5;
  const double v_min = detection2d_msg.bbox.center.position.y - detection2d_msg.bbox.size_y * 0.5;
  const double v_max = detection2d_msg.bbox.center.position.y + detection2d_msg.bbox.size_y * 0.5;

  for (const auto& point : cloud->points)
  {
    if (point.z < 0.0)
      continue;

    cv::Point2d u_v;
    if (!project3DToPixelRectified(camera_info, cv::Point3d(point.x, point.y, point.z), u_v))
      continue;

    if (u_v.x >= u_min && u_v.x <= u_max && u_v.y >= v_min && u_v.y <= v_max)
    {
      raw_detection_cloud->points.push_back(point);
    }
  }
}

void CameraLidarPerceptionNode::createBoundingBox(
    vision_msgs::msg::Detection3DArray& detection3d_array_msg, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<vision_msgs::msg::ObjectHypothesisWithPose>& detections_results_msg)
{
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);

  double theta = -std::atan2(centroid[1], std::sqrt(std::pow(centroid[0], 2) + std::pow(centroid[2], 2)));
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()));

  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  transformPointCloud(cloud, rotated_cloud, transform);

  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*rotated_cloud, min_pt, max_pt);

  Eigen::Vector4f transformed_bbox_center =
      Eigen::Vector4f((min_pt.x + max_pt.x) / 2, (min_pt.y + max_pt.y) / 2, (min_pt.z + max_pt.z) / 2, 1.0);

  Eigen::Vector4f bbox_center = transform.inverse() * transformed_bbox_center;
  Eigen::Quaternionf q(transform.inverse().rotation());

  vision_msgs::msg::Detection3D detection3d_msg;
  detection3d_msg.bbox.center.position.x = bbox_center[0];
  detection3d_msg.bbox.center.position.y = bbox_center[1];
  detection3d_msg.bbox.center.position.z = bbox_center[2];
  detection3d_msg.bbox.center.orientation.x = q.x();
  detection3d_msg.bbox.center.orientation.y = q.y();
  detection3d_msg.bbox.center.orientation.z = q.z();
  detection3d_msg.bbox.center.orientation.w = q.w();
  detection3d_msg.bbox.size.x = max_pt.x - min_pt.x;
  detection3d_msg.bbox.size.y = max_pt.y - min_pt.y;
  detection3d_msg.bbox.size.z = max_pt.z - min_pt.z;
  detection3d_msg.results = detections_results_msg;
  detection3d_array_msg.detections.push_back(detection3d_msg);
}

// Inspired from here: https://pcl.readthedocs.io/projects/tutorials/en/master/voxel_grid.html
pcl::PointCloud<pcl::PointXYZ>::Ptr
CameraLidarPerceptionNode::downSampleCloudMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud_msg, *cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ>::Ptr voxel_grid(new pcl::VoxelGrid<pcl::PointXYZ>);
  voxel_grid->setInputCloud(cloud);
  voxel_grid->setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
  voxel_grid->filter(*downsampled_cloud);

  return downsampled_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CameraLidarPerceptionNode::cloud2TransformedCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& source_frame, const std::string& target_frame,
    const rclcpp::Time& stamp)
{
  try
  {
    geometry_msgs::msg::TransformStamped tf_stamp =
        tf2_buffer_->lookupTransform(target_frame, source_frame, stamp, 100ms);
    Eigen::Affine3f eigen_transform = tf2::transformToEigen(tf_stamp.transform).cast<float>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    transformPointCloud(cloud, transformed_cloud, eigen_transform);

    return transformed_cloud;
  }
  catch (tf2::TransformException& e)
  {
    RCLCPP_WARN(this->get_logger(), "%s", e.what());
    return cloud;
  }
}

// Inspired from this tutorial: https://pcl.readthedocs.io/projects/tutorials/en/master/cluster_extraction.html
pcl::PointCloud<pcl::PointXYZ>::Ptr
CameraLidarPerceptionNode::eucludianClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(kd_tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  float min_distance = std::numeric_limits<float>::max();
  pcl::PointCloud<pcl::PointXYZ>::Ptr closest_cluster(new pcl::PointCloud<pcl::PointXYZ>);

  for (const auto& cluster : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& indice : cluster.indices)
    {
      cloud_cluster->push_back((*cloud)[indice]);
    }

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_cluster, centroid);
    float distance = centroid.norm();

    if (distance < min_distance)
    {
      min_distance = distance;
      *closest_cluster = *cloud_cluster;
    }
  }

  return closest_cluster;
}

visualization_msgs::msg::MarkerArray CameraLidarPerceptionNode::createMarkerArray(
    const vision_msgs::msg::Detection3DArray& detection3d_array_msg, const double& duration)
{
  return visualization_msgs::msg::MarkerArray();
}

}  // namespace camera_lidar_3d_perception

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(camera_lidar_3d_perception::CameraLidarPerceptionNode)