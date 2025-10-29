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
  detection_3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("detections_3d", 10);
  marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("detection_markers", 10);

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

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud = downSampleCloudMsg(pointcloud_msg);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_base =
      cloud2TransformedCloud(downsampled_cloud, pointcloud_msg->header.frame_id, "base_link", stamp);

  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_cam =
      cloud2TransformedCloud(cloud_base, "base_link", camera_optical_frame_, stamp);

  vision_msgs::msg::Detection3DArray detection3d_array_msg;
  sensor_msgs::msg::PointCloud2 detection_cloud_msg;

  projectCloud(point_cloud_cam, detections_msg, camera_info_msg, pointcloud_msg->header, detection3d_array_msg,
               detection_cloud_msg);

  visualization_msgs::msg::MarkerArray marker_array_msg =
      createMarkerArray(detection3d_array_msg, current_time.seconds());

  marker_array_pub_->publish(marker_array_msg);
  detection_3d_pub_->publish(detection3d_array_msg);
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
    if (std::abs(pt.x) > 40.0)
      continue;
    if (std::abs(pt.y) > 20.0)
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
    return;

  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*cloud_in_target_frame, min_pt, max_pt);

  const double cx = 0.5 * (static_cast<double>(min_pt.x) + static_cast<double>(max_pt.x));
  const double cy = 0.5 * (static_cast<double>(min_pt.y) + static_cast<double>(max_pt.y));
  const double cz = 0.5 * (static_cast<double>(min_pt.z) + static_cast<double>(max_pt.z));

  const double sx = std::max(0.0, static_cast<double>(max_pt.x) - static_cast<double>(min_pt.x));
  const double sy = std::max(0.0, static_cast<double>(max_pt.y) - static_cast<double>(min_pt.y));
  const double sz = std::max(0.0, static_cast<double>(max_pt.z) - static_cast<double>(min_pt.z));

  vision_msgs::msg::Detection3D det;
  det.bbox.center.position.x = cx;
  det.bbox.center.position.y = cy;
  det.bbox.center.position.z = cz;
  det.bbox.center.orientation.x = 0.0;
  det.bbox.center.orientation.y = 0.0;
  det.bbox.center.orientation.z = 0.0;
  det.bbox.center.orientation.w = 1.0;

  det.bbox.size.x = sx;
  det.bbox.size.y = sy;
  det.bbox.size.z = sz;

  det.results = detections_results_msg;

  detection3d_array_msg.detections.push_back(det);
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
        tf2_buffer_->lookupTransform(target_frame, source_frame, stamp, 100ms);
    Eigen::Affine3f T = tf2::transformToEigen(tf_stamp.transform).cast<float>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    transformPointCloud(cloud, out, T);
    return out;
  }
  catch (const tf2::TransformException& e)
  {
    RCLCPP_WARN(this->get_logger(), "TF lookup %s->%s failed: %s", source_frame.c_str(), target_frame.c_str(),
                e.what());
    return cloud;  // return as-is to avoid dropping data
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

  // Choose closest cluster to origin of target frame
  float min_dist = std::numeric_limits<float>::max();
  pcl::PointCloud<pcl::PointXYZ>::Ptr best(new pcl::PointCloud<pcl::PointXYZ>);

  for (const auto& idx : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr clu(new pcl::PointCloud<pcl::PointXYZ>);
    clu->reserve(idx.indices.size());
    for (int i : idx.indices)
      clu->push_back((*cloud)[i]);

    Eigen::Vector4f c;
    pcl::compute3DCentroid(*clu, c);
    float d = c.head<3>().norm();
    if (d < min_dist)
    {
      min_dist = d;
      *best = *clu;
    }
  }

  return best;
}

visualization_msgs::msg::MarkerArray CameraLidarPerceptionNode::createMarkerArray(
    const vision_msgs::msg::Detection3DArray& detection3d_array_msg, const double& duration)
{
  visualization_msgs::msg::MarkerArray marker_array_msg;

  for (size_t i = 0; i < detection3d_array_msg.detections.size(); i++)
  {
    if (std::isfinite(detection3d_array_msg.detections[i].bbox.size.x) &&
        std::isfinite(detection3d_array_msg.detections[i].bbox.size.y) &&
        std::isfinite(detection3d_array_msg.detections[i].bbox.size.z))
    {
      visualization_msgs::msg::Marker marker_msg;
      marker_msg.header = detection3d_array_msg.header;
      marker_msg.ns = "detection";
      marker_msg.id = i;
      marker_msg.type = visualization_msgs::msg::Marker::CUBE;
      marker_msg.action = visualization_msgs::msg::Marker::ADD;
      marker_msg.pose = detection3d_array_msg.detections[i].bbox.center;
      marker_msg.scale.x = detection3d_array_msg.detections[i].bbox.size.x;
      marker_msg.scale.y = detection3d_array_msg.detections[i].bbox.size.y;
      marker_msg.scale.z = detection3d_array_msg.detections[i].bbox.size.z;
      marker_msg.color.r = 0.0;
      marker_msg.color.g = 1.0;
      marker_msg.color.b = 0.0;
      marker_msg.color.a = 0.5;
      marker_msg.lifetime = rclcpp::Duration(std::chrono::duration<double>(duration));
      marker_array_msg.markers.push_back(marker_msg);
    }
  }

  return marker_array_msg;
}

}  // namespace camera_lidar_3d_perception

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(camera_lidar_3d_perception::CameraLidarPerceptionNode)