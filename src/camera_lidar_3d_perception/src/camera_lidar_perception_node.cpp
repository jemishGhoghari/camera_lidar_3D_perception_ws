#include "camera_lidar_3d_perception/camera_lidar_perception_node.hpp"

namespace camera_lidar_3d_perception
{

CameraLidarPerceptionNode::CameraLidarPerceptionNode(const rclcpp::NodeOptions& options)
  : Node("camera_lidar_3d_node", options)
{
  voxel_leaf_size_ = this->declare_parameter<double>("voxel_leaf_size", 0.15);      // was 0.5
  cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.40);  // was 0.5
  min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 10);         // was 100
  max_cluster_size_ = this->declare_parameter<int>("max_cluster_size", 25000);
  camera_optical_frame_ = this->declare_parameter<std::string>("camera_optical_frame", "zed_left_camera_optical_frame");
  target_frame_ = this->declare_parameter<std::string>("target_frame", "map");

  // Subscribers
  pointcloud_sub_.subscribe(this, "pointcloud");
  detection_2d_sub_.subscribe(this, "detections");
  camera_info_sub_.subscribe(this, "camera_info");

  // Approximate sync
  sync_ = std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(10);
  sync_->connectInput(pointcloud_sub_, detection_2d_sub_, camera_info_sub_);
  sync_->registerCallback(std::bind(&CameraLidarPerceptionNode::syncCallback, this, std::placeholders::_1,
                                    std::placeholders::_2, std::placeholders::_3));

  // Publishers
  detection_3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("detections_3d", 50);
  marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("detection_markers", 50);

  // TF
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
  rclcpp::Duration time_diff = current_time - last_callback_time_;
  last_callback_time_ = current_time;

  const auto stamp = pointcloud_msg->header.stamp;

  if (detections_msg->detections.empty())
  {
    RCLCPP_DEBUG(this->get_logger(), "No 2D detections received, skipping frame");
    return;
  }

  RCLCPP_DEBUG(this->get_logger(), "Received %zu 2D detections", detections_msg->detections.size());

  if (!tf2_buffer_->canTransform(camera_optical_frame_, pointcloud_msg->header.frame_id, stamp, 100ms))
  {
    std::string error_string;
    bool latest_available = tf2_buffer_->canTransform(camera_optical_frame_, pointcloud_msg->header.frame_id,
                                                      tf2::TimePointZero, 100ms, &error_string);

    if (latest_available)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Transform %s->%s not available, but available at latest. "
                           "Consider using tf2::TimePointZero or check time sync.",
                           pointcloud_msg->header.frame_id.c_str(), camera_optical_frame_.c_str());
    }
    else
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Transform %s->%s not available: %s",
                           pointcloud_msg->header.frame_id.c_str(), camera_optical_frame_.c_str(),
                           error_string.c_str());
    }
    return;
  }

  if (!tf2_buffer_->canTransform(target_frame_, camera_optical_frame_, stamp, 100ms))
  {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Transform %s->%s not available",
                         camera_optical_frame_.c_str(), target_frame_.c_str());
    return;
  }

  // Step 1: Downsample the incoming point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud = downSampleCloudMsg(pointcloud_msg);

  if (!downsampled_cloud || downsampled_cloud->empty())
  {
    RCLCPP_WARN(this->get_logger(), "Downsampled cloud is empty!");
    return;
  }

  RCLCPP_DEBUG(
      this->get_logger(), "Point cloud: original=%u, downsampled=%lu (reduction: %.1f%%)",
      pointcloud_msg->width * pointcloud_msg->height, downsampled_cloud->size(),
      100.0 * (1.0 - (double)downsampled_cloud->size() / (double)(pointcloud_msg->width * pointcloud_msg->height)));

  // Step 2: Transform DIRECTLY from LiDAR frame to camera optical frame
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_cam =
      cloud2TransformedCloud(downsampled_cloud,
                             pointcloud_msg->header.frame_id,  // Source: LiDAR frame
                             camera_optical_frame_,            // Target: Camera frame
                             stamp);

  if (!point_cloud_cam || point_cloud_cam->empty())
  {
    RCLCPP_WARN(this->get_logger(), "Transformed cloud (camera frame) is empty after transform from %s to %s!",
                pointcloud_msg->header.frame_id.c_str(), camera_optical_frame_.c_str());
    return;
  }

  RCLCPP_DEBUG(this->get_logger(), "Transformed %zu points to camera frame", point_cloud_cam->size());

  vision_msgs::msg::Detection3DArray detection3d_array_msg;
  detection3d_array_msg.header.frame_id = target_frame_;
  detection3d_array_msg.header.stamp = stamp;

  sensor_msgs::msg::PointCloud2 detection_cloud_msg;

  // Process each 2D detection
  int successful_detections = 0;
  int failed_detections = 0;

  for (size_t i = 0; i < detections_msg->detections.size(); ++i)
  {
    const auto& detection_2d = detections_msg->detections[i];

    RCLCPP_DEBUG(this->get_logger(), "Processing detection %zu: bbox center=(%.1f, %.1f), size=(%.1f x %.1f)", i,
                 detection_2d.bbox.center.position.x, detection_2d.bbox.center.position.y, detection_2d.bbox.size_x,
                 detection_2d.bbox.size_y);

    // Step 4a: Extract points within 2D bounding box (in camera frame)
    pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
    processPointsWithBbox(point_cloud_cam, detection_2d, camera_info_msg, detection_cloud_raw);

    if (detection_cloud_raw->empty())
    {
      RCLCPP_DEBUG(this->get_logger(), "Detection %zu: No points found within 2D bbox", i);
      failed_detections++;
      continue;
    }

    RCLCPP_DEBUG(this->get_logger(), "Detection %zu: Found %zu points in 2D bbox", i, detection_cloud_raw->size());

    pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud_in_target =
        cloud2TransformedCloud(detection_cloud_raw,
                               camera_optical_frame_,  // Source: camera frame
                               target_frame_,          // Target: map/world frame
                               stamp);

    if (detection_cloud_in_target->empty())
    {
      RCLCPP_DEBUG(this->get_logger(), "Detection %zu: Cloud empty after transform to target frame", i);
      failed_detections++;
      continue;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_detection_cloud = eucludianClustering(detection_cloud_in_target);

    if (clustered_detection_cloud->empty())
    {
      RCLCPP_DEBUG(this->get_logger(), "Detection %zu: No valid clusters found (tried clustering %zu points)", i,
                   detection_cloud_in_target->size());
      failed_detections++;
      continue;
    }

    RCLCPP_DEBUG(this->get_logger(), "Detection %zu: Clustered to %zu points", i, clustered_detection_cloud->size());

    createBoundingBox(detection3d_array_msg, clustered_detection_cloud, detection_2d.results);

    successful_detections++;
  }

  RCLCPP_DEBUG(this->get_logger(), "Detection results: %d successful, %d failed out of %zu 2D detections",
               successful_detections, failed_detections, detections_msg->detections.size());

  if (detection3d_array_msg.detections.empty())
  {
    RCLCPP_DEBUG(this->get_logger(), "No valid 3D detections generated");

    visualization_msgs::msg::MarkerArray empty_markers;
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = target_frame_;
    delete_marker.header.stamp = stamp;
    delete_marker.ns = "detection";
    delete_marker.id = 0;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    empty_markers.markers.push_back(delete_marker);
    marker_array_pub_->publish(empty_markers);

    return;
  }

  visualization_msgs::msg::MarkerArray marker_array_msg = createMarkerArray(detection3d_array_msg, 2.0);

  marker_array_pub_->publish(marker_array_msg);
  detection_3d_pub_->publish(detection3d_array_msg);

  RCLCPP_DEBUG(this->get_logger(), "Published %zu 3D detections and markers", detection3d_array_msg.detections.size());
}

void CameraLidarPerceptionNode::transformPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                                    const Eigen::Affine3f& transform)
{
  const int n = static_cast<int>(cloud_in->size());
  cloud_out->resize(n);
  for (int i = 0; i < n; ++i)
  {
    const auto& p = cloud_in->points[i];
    cloud_out->points[i].x = transform(0, 0) * p.x + transform(0, 1) * p.y + transform(0, 2) * p.z + transform(0, 3);
    cloud_out->points[i].y = transform(1, 0) * p.x + transform(1, 1) * p.y + transform(1, 2) * p.z + transform(1, 3);
    cloud_out->points[i].z = transform(2, 0) * p.x + transform(2, 1) * p.y + transform(2, 2) * p.z + transform(2, 3);
  }
}

void CameraLidarPerceptionNode::projectCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_cam,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& yolo_detections_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg, const std_msgs::msg::Header& header,
    vision_msgs::msg::Detection3DArray& detection3d_array_msg,
    sensor_msgs::msg::PointCloud2& combine_detection_cloud_msg)
{
  const auto stamp = header.stamp;

  detection3d_array_msg.header.frame_id = target_frame_;
  detection3d_array_msg.header.stamp = stamp;

  pcl::PointCloud<pcl::PointXYZ>::Ptr combine_detection_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t i = 0; i < yolo_detections_msg->detections.size(); ++i)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
    processPointsWithBbox(point_cloud_cam, yolo_detections_msg->detections[i], camera_info_msg, detection_cloud_raw);

    if (detection_cloud_raw->empty())
      continue;

    pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud_in_target =
        cloud2TransformedCloud(detection_cloud_raw, camera_optical_frame_, target_frame_, stamp);

    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_detection_cloud = eucludianClustering(detection_cloud_in_target);
    if (clustered_detection_cloud->empty())
      continue;
    *combine_detection_cloud += *clustered_detection_cloud;

    createBoundingBox(detection3d_array_msg, combine_detection_cloud, yolo_detections_msg->detections[i].results);
  }
}

bool CameraLidarPerceptionNode::project3DToPixelRectified(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cam_info_msg, const cv::Point3d& p3d, cv::Point2d& uv)
{
  const double fx = cam_info_msg->k[0];
  const double fy = cam_info_msg->k[4];
  const double cx = cam_info_msg->k[2];
  const double cy = cam_info_msg->k[5];

  if (p3d.z <= 0.0)
    return false;

  const double invZ = 1.0 / p3d.z;
  uv.x = fx * (p3d.x * invZ) + cx;
  uv.y = fy * (p3d.y * invZ) + cy;
  return true;
}

void CameraLidarPerceptionNode::processPointsWithBbox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_cam,
                                                      const vision_msgs::msg::Detection2D& detection2d_msg,
                                                      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info,
                                                      pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_detection_cloud)
{
  const double u_min = detection2d_msg.bbox.center.position.x - detection2d_msg.bbox.size_x * 0.5;
  const double u_max = detection2d_msg.bbox.center.position.x + detection2d_msg.bbox.size_x * 0.5;
  const double v_min = detection2d_msg.bbox.center.position.y - detection2d_msg.bbox.size_y * 0.5;
  const double v_max = detection2d_msg.bbox.center.position.y + detection2d_msg.bbox.size_y * 0.5;

  for (const auto& pt : cloud_cam->points)
  {
    if (pt.z <= 0.0)
      continue;
    if (pt.z > 60.0)
      continue;
    if (std::abs(pt.x) > 80.0)
      continue;
    if (std::abs(pt.y) > 40.0)
      continue;

    cv::Point2d uv;
    if (!project3DToPixelRectified(camera_info, cv::Point3d(pt.x, pt.y, pt.z), uv))
      continue;

    if (uv.x >= u_min && uv.x <= u_max && uv.y >= v_min && uv.y <= v_max)
      raw_detection_cloud->points.push_back(pt);
  }
}

void CameraLidarPerceptionNode::createBoundingBox(
    vision_msgs::msg::Detection3DArray& detection3d_array_msg,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in_target_frame,
    const std::vector<vision_msgs::msg::ObjectHypothesisWithPose>& detections_results_msg)
{
  if (!cloud_in_target_frame || cloud_in_target_frame->empty())
  {
    RCLCPP_WARN(this->get_logger(), "Cannot create bbox: empty point cloud");
    return;
  }

  // Get min/max points
  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*cloud_in_target_frame, min_pt, max_pt);

  // Calculate dimensions
  const double size_x = std::max(0.01, static_cast<double>(max_pt.x - min_pt.x));
  const double size_y = std::max(0.01, static_cast<double>(max_pt.y - min_pt.y));
  const double size_z = std::max(0.01, static_cast<double>(max_pt.z - min_pt.z));

  // CRITICAL FIX #1: Ground-aligned bounding box
  // Instead of centering in Z, align the bottom of the box with min_pt.z
  const double center_x = 0.5 * (static_cast<double>(min_pt.x) + static_cast<double>(max_pt.x));
  const double center_y = 0.5 * (static_cast<double>(min_pt.y) + static_cast<double>(max_pt.y));
  const double center_z = static_cast<double>(min_pt.z) + 0.5 * size_z;  // Bottom + half height

  RCLCPP_DEBUG(this->get_logger(),
               "BBox: center=(%.2f, %.2f, %.2f), size=(%.2f, %.2f, %.2f), "
               "z_range=[%.2f, %.2f]",
               center_x, center_y, center_z, size_x, size_y, size_z, min_pt.z, max_pt.z);

  // Check for reasonable dimensions
  if (size_x > 50.0 || size_y > 50.0 || size_z > 50.0)
  {
    RCLCPP_WARN(this->get_logger(), "Unreasonably large bbox detected (%.2f x %.2f x %.2f), skipping", size_x, size_y,
                size_z);
    return;
  }

  // Check for floating boxes (too high above ground)
  // Assuming target_frame is "map" or similar world frame
  if (min_pt.z > 2.0)  // If bottom is more than 2m above ground
  {
    RCLCPP_WARN(this->get_logger(), "Floating bbox detected (min_z=%.2f), possible transform issue", min_pt.z);
  }

  // CRITICAL FIX #2: Use PCA for better orientation estimation
  // This helps align the box with the object's principal axes
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud_in_target_frame, centroid);

  Eigen::Matrix3f covariance;
  pcl::computeCovarianceMatrixNormalized(*cloud_in_target_frame, centroid, covariance);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();

  // Ensure right-handed coordinate system
  eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));

  // Convert rotation matrix to quaternion
  Eigen::Quaternionf quat(eigen_vectors);
  quat.normalize();

  // CRITICAL FIX #3: For ground-based objects, force upright orientation
  // This prevents boxes from tilting due to noise in the point cloud
  // Only keep yaw rotation, reset pitch and roll
  double roll, pitch, yaw;
  tf2::Quaternion tf_quat(quat.x(), quat.y(), quat.z(), quat.w());
  tf2::Matrix3x3(tf_quat).getRPY(roll, pitch, yaw);

  // Create upright quaternion (only yaw rotation)
  tf2::Quaternion upright_quat;
  upright_quat.setRPY(0.0, 0.0, yaw);  // Zero roll and pitch
  upright_quat.normalize();

  RCLCPP_DEBUG(this->get_logger(), "Orientation: original RPY=(%.2f, %.2f, %.2f), corrected=(0.0, 0.0, %.2f)",
               roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI, yaw * 180.0 / M_PI);

  // Create detection message
  vision_msgs::msg::Detection3D det;

  // Position - use ground-aligned center
  det.bbox.center.position.x = center_x;
  det.bbox.center.position.y = center_y;
  det.bbox.center.position.z = center_z;

  // Orientation - use upright quaternion
  det.bbox.center.orientation.x = upright_quat.x();
  det.bbox.center.orientation.y = upright_quat.y();
  det.bbox.center.orientation.z = upright_quat.z();
  det.bbox.center.orientation.w = upright_quat.w();

  // Size
  det.bbox.size.x = size_x;
  det.bbox.size.y = size_y;
  det.bbox.size.z = size_z;

  // Copy detection results (class labels, scores, etc.)
  det.results = detections_results_msg;

  // Validate before adding
  if (std::isfinite(center_x) && std::isfinite(center_y) && std::isfinite(center_z) && std::isfinite(size_x) &&
      std::isfinite(size_y) && std::isfinite(size_z))
  {
    detection3d_array_msg.detections.push_back(det);
  }
  else
  {
    RCLCPP_WARN(this->get_logger(), "Non-finite values in bbox, skipping");
  }
}

visualization_msgs::msg::MarkerArray CameraLidarPerceptionNode::createMarkerArray(
    const vision_msgs::msg::Detection3DArray& detection3d_array_msg, const double& duration)
{
  visualization_msgs::msg::MarkerArray marker_array_msg;

  visualization_msgs::msg::Marker delete_marker;
  delete_marker.header = detection3d_array_msg.header;
  delete_marker.ns = "detection";
  delete_marker.id = 0;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array_msg.markers.push_back(delete_marker);

  for (size_t i = 0; i < detection3d_array_msg.detections.size(); i++)
  {
    const auto& detection = detection3d_array_msg.detections[i];

    if (!std::isfinite(detection.bbox.size.x) || !std::isfinite(detection.bbox.size.y) ||
        !std::isfinite(detection.bbox.size.z) || !std::isfinite(detection.bbox.center.position.x) ||
        !std::isfinite(detection.bbox.center.position.y) || !std::isfinite(detection.bbox.center.position.z))
    {
      RCLCPP_WARN(this->get_logger(), "Skipping marker %zu: non-finite values", i);
      continue;
    }

    if (detection.bbox.size.x < 0.01 || detection.bbox.size.y < 0.01 || detection.bbox.size.z < 0.01)
    {
      RCLCPP_DEBUG(this->get_logger(), "Skipping marker %zu: too small", i);
      continue;
    }

    visualization_msgs::msg::Marker box_marker;
    box_marker.header = detection3d_array_msg.header;
    box_marker.ns = "detection";
    box_marker.id = i * 3 + 1;  // Offset IDs to avoid conflicts
    box_marker.type = visualization_msgs::msg::Marker::CUBE;
    box_marker.action = visualization_msgs::msg::Marker::ADD;

    box_marker.pose = detection.bbox.center;
    box_marker.scale = detection.bbox.size;

    box_marker.color.r = 0.0;
    box_marker.color.g = 1.0;
    box_marker.color.b = 0.0;
    box_marker.color.a = 0.3;

    box_marker.lifetime = rclcpp::Duration::from_seconds(duration);
    marker_array_msg.markers.push_back(box_marker);

    visualization_msgs::msg::Marker edge_marker;
    edge_marker.header = detection3d_array_msg.header;
    edge_marker.ns = "detection";
    edge_marker.id = i * 3 + 2;
    edge_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    edge_marker.action = visualization_msgs::msg::Marker::ADD;

    edge_marker.pose = detection.bbox.center;
    edge_marker.scale.x = 0.02;

    edge_marker.color.r = 0.0;
    edge_marker.color.g = 1.0;
    edge_marker.color.b = 0.0;
    edge_marker.color.a = 1.0;

    const double hx = detection.bbox.size.x * 0.5;
    const double hy = detection.bbox.size.y * 0.5;
    const double hz = detection.bbox.size.z * 0.5;

    std::vector<geometry_msgs::msg::Point> corners(8);
    corners[0].x = -hx;
    corners[0].y = -hy;
    corners[0].z = -hz;
    corners[1].x = hx;
    corners[1].y = -hy;
    corners[1].z = -hz;
    corners[2].x = hx;
    corners[2].y = hy;
    corners[2].z = -hz;
    corners[3].x = -hx;
    corners[3].y = hy;
    corners[3].z = -hz;
    corners[4].x = -hx;
    corners[4].y = -hy;
    corners[4].z = hz;
    corners[5].x = hx;
    corners[5].y = -hy;
    corners[5].z = hz;
    corners[6].x = hx;
    corners[6].y = hy;
    corners[6].z = hz;
    corners[7].x = -hx;
    corners[7].y = hy;
    corners[7].z = hz;

    int edges[12][2] = {
      { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },  // Bottom face
      { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },  // Top face
      { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }   // Vertical edges
    };

    for (int e = 0; e < 12; e++)
    {
      edge_marker.points.push_back(corners[edges[e][0]]);
      edge_marker.points.push_back(corners[edges[e][1]]);
    }

    edge_marker.lifetime = rclcpp::Duration::from_seconds(duration);
    marker_array_msg.markers.push_back(edge_marker);

    if (!detection.results.empty())
    {
      visualization_msgs::msg::Marker text_marker;
      text_marker.header = detection3d_array_msg.header;
      text_marker.ns = "detection";
      text_marker.id = i * 3 + 3;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;

      text_marker.pose.position.x = detection.bbox.center.position.x;
      text_marker.pose.position.y = detection.bbox.center.position.y;
      text_marker.pose.position.z = detection.bbox.center.position.z + detection.bbox.size.z * 0.5 + 0.3;
      text_marker.pose.orientation.w = 1.0;

      text_marker.scale.z = 0.3;

      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 1.0;
      text_marker.color.a = 1.0;

      const auto& result = detection.results[0];
      text_marker.text =
          result.hypothesis.class_id + " (" + std::to_string(static_cast<int>(result.hypothesis.score * 100)) + "%)";

      text_marker.lifetime = rclcpp::Duration::from_seconds(duration);
      marker_array_msg.markers.push_back(text_marker);
    }
  }

  RCLCPP_DEBUG(this->get_logger(), "Created %zu markers", marker_array_msg.markers.size());
  return marker_array_msg;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
CameraLidarPerceptionNode::downSampleCloudMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud_msg, *cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
  vg.filter(*downsampled);
  return downsampled;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CameraLidarPerceptionNode::cloud2TransformedCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& source_frame, const std::string& target_frame,
    const rclcpp::Time& stamp)
{
  try
  {
    geometry_msgs::msg::TransformStamped tf_stamp =
        tf2_buffer_->lookupTransform(target_frame, source_frame, stamp, 500ms);
    Eigen::Affine3f T = tf2::transformToEigen(tf_stamp.transform).cast<float>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    transformPointCloud(cloud, out, T);
    return out;
  }
  catch (const tf2::TransformException& e)
  {
    RCLCPP_WARN(this->get_logger(), "TF lookup %s->%s failed: %s", source_frame.c_str(), target_frame.c_str(),
                e.what());
    return cloud;
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
CameraLidarPerceptionNode::eucludianClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  if (!cloud || cloud->empty())
  {
    return pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  }

  pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(kd_tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  size_t max_size = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr best(new pcl::PointCloud<pcl::PointXYZ>);

  for (const auto& idx : cluster_indices)
  {
    if (idx.indices.size() > max_size)
    {
      max_size = idx.indices.size();
      best->clear();
      best->reserve(idx.indices.size());
      for (int i : idx.indices)
        best->push_back((*cloud)[i]);
    }
  }

  return best;
}

}  // namespace camera_lidar_3d_perception

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(camera_lidar_3d_perception::CameraLidarPerceptionNode)