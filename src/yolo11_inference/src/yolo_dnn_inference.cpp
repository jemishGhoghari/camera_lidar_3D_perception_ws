#include "yolo11_inference/yolo_dnn_inference.hpp"

namespace yolo11_inference
{

Yolo11DNNInference::Yolo11DNNInference(const YoloDNNConfig& config, std::vector<std::string> class_names,
                                       const rclcpp::Logger& logger)
  : config_(config), logger_(logger), class_names_(std::move(class_names))
{
  coco_classes_ = loadCOCOClasses();
  // Check if the model is available
  if (config_.modelPath.empty())
  {
    RCLCPP_FATAL(logger_, "ONNX model path is empty");
  }

  // Check if all classes should be detected OR certain classes only
  if (class_names_.empty())
  {
    RCLCPP_WARN(logger_, "Class names list is empty. All classes will be detected.");
    class_names_ = coco_classes_;
  }

  // Load network from ONNX model
  net_ = cv::dnn::readNetFromONNX(config_.modelPath);
  if (net_.empty())
  {
    RCLCPP_FATAL(logger_, "Failed to load ONNX model from path: %s", config_.modelPath.c_str());
  }

  // Set preferable backend on the system capabilities and configuration
  try
  {
    if (config_.use_cuda)
    {
#ifdef HAVE_CUDA  // found it here: https://forum.opencv.org/t/determine-if-opencv-was-built-with-support-for-cuda/5716
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
      RCLCPP_INFO(logger_, "Using CUDA for DNN inference");
#else
      RCLCPP_WARN(logger_, "OpenCV is not built with CUDA support. Falling back to CPU.");
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
    }
    else
    {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      RCLCPP_INFO(logger_, "Using CPU for DNN inference");
    }
  }
  catch (const cv::Exception& e)
  {
    RCLCPP_FATAL(logger_, "Error setting DNN backend/target: %s", e.what());
  }

  // Warm up the model with a dummy inference. Why we need this? this can help understand:
  // https://github.com/ultralytics/ultralytics/issues/11238
  cv::Mat dummy_input(config_.networkInputHeight, config_.networkInputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::Mat blob =
      cv::dnn::blobFromImage(dummy_input, 1.0 / 255.0, cv::Size(config_.networkInputWidth, config_.networkInputHeight),
                             cv::Scalar(), true, false);
  net_.setInput(blob);
  net_.forward();
  RCLCPP_INFO(logger_, "DNN model loaded and warmed up successfully");
}

Yolo11DNNInference::~Yolo11DNNInference()
{
}

bool Yolo11DNNInference::infer(const cv::Mat& input_image, DetectionResult& result)
{
  result = DetectionResult{};
  if (input_image.empty())  // Ensure image is not empty
    return false;

  cv::Vec4i pad{ 0, 0, 0, 0 };
  float scale = 1.f;

  // Letterbox image
  cv::Mat letterboxed_image = letterbox(input_image, config_.networkInputWidth, config_.networkInputHeight, pad, scale);

  // Create blob from input image
  cv::Mat blob = cv::dnn::blobFromImage(letterboxed_image, 1.0 / 255.0,
                                        cv::Size(config_.networkInputWidth, config_.networkInputHeight), cv::Scalar(),
                                        /*SwapRB*/ true, /*Crop*/ false);

  net_.setInput(blob);

  std::vector<cv::Mat> outs;
  try
  {
    // forward pass through the model to get the detections
    net_.forward(outs);
  }
  catch (const cv::Exception& e)
  {
    RCLCPP_WARN(logger_, "DNN Forward pass failed: %s", e.what());
  }

  cv::Mat raw_output;

  if (!outs.empty())
  {
    raw_output = outs[0];  // Use the first model output tensor
  }
  else
  {
    raw_output = net_.forward();  // fallback
  }

  // Now pass a single tensor to toNxC
  cv::Mat det = toNxC(raw_output);  // det = (N × C)

  // validate the output dimentions
  if (det.cols < 4)
  {
    RCLCPP_WARN(logger_, "Bad shape: %d x %d", det.dims, det.cols);
    return false;
  }

  const int C = det.cols;  // extract field size
  const int K = C - 4;     // extract the number of classes

  RCLCPP_DEBUG(logger_, "K inferred as %d", K);

  if (K <= 0)
  {
    RCLCPP_WARN(logger_, "Invalid class count inferred (K=%d).", K);
    return false;
  }

  // Decode the detections from the DNN outputs
  std::vector<cv::Rect> boxes;
  std::vector<float> confidence_scores;
  std::vector<int> class_ids;

  try
  {
    decodeDetections(det, static_cast<float>(config_.confThreshold), K, input_image.cols, input_image.rows, pad, scale,
                     boxes, confidence_scores, class_ids);
  }
  catch (const cv::Exception& e)
  {
    RCLCPP_WARN(logger_, "Decode failed: %s", e.what());
    return false;
  }

  // Apply Non-Max-Supression to get the best bbox out of large number of bounding boxes
  std::vector<int> keep_indices;
  cv::dnn::NMSBoxes(boxes, confidence_scores, config_.confThreshold, config_.nmsThreshold, keep_indices);

  for (int idx : keep_indices)
  {
    if (idx < 0 || idx >= (int)boxes.size())
      continue;

    int cid = (idx < (int)class_ids.size()) ? class_ids[idx] : 0;
    if (cid < 0 || cid > K)
      continue;

    result.boxes.push_back(boxes[idx]);
    result.confidence_scores.push_back(confidence_scores[idx]);
    result.class_ids.push_back(cid);
  }

  return true;
}

void Yolo11DNNInference::decodeDetections(const cv::Mat& output, float conf_threshold, int K, int img_width,
                                          int img_height, const cv::Vec4i& pad, float scale,
                                          std::vector<cv::Rect>& boxes, std::vector<float>& confidence_scores,
                                          std::vector<int>& class_ids)
{
  boxes.clear();
  confidence_scores.clear();
  class_ids.clear();

  CV_Assert(output.dims == 2);
  // Yolo detection head structure is (batch_size, detecion fields, num of detected boxes) -> (1, 84, 8400)
  // NOTE: It may change depending the model's input size
  const int N = output.rows;  // (center coor.) + (hight, wifth) + confidence of each class = 4 + 80 (Yolo detection
                              // head output structure)
  const int C = output.cols;  // number of detected boxes
  // RCLCPP_WARN(logger_, "Unexpected detection field size: %d (expected %d)", C, 4 + K);
  CV_Assert(C == 4 + K);  // 4 is not a magic number. 4 represents (center coor. (X, Y)) + (hight, width)

  // remove padding to get original image -> remove padding and scale down to original
  auto unletterbox = [&](cv::Rect2f r) -> cv::Rect2f {
    r.x -= pad[0];
    r.y -= pad[1];
    r.x /= scale;
    r.y /= scale;
    r.width /= scale;
    r.height /= scale;
    return r;
  };

  for (int i = 0; i < N; ++i)
  {
    const float* field_pointer = output.ptr<float>(i);
    float center_x = field_pointer[0];
    float center_y = field_pointer[1];
    float box_width = field_pointer[2];
    float box_height = field_pointer[3];

    // extract class with best confidence score
    int best_class = 0;
    float best_score = field_pointer[4];
    for (int c = 1; c < K; ++c)
    {
      float score = field_pointer[4 + c];
      if (score > best_score)
      {
        best_score = score;
        best_class = c;
      }
    }

    // filter out the class names from the COCO dataset labels
    const std::string& label_name = coco_classes_.at(best_class);
    auto it = std::find_if(class_names_.begin(), class_names_.end(),
                           [&](const std::string& name) { return name == label_name; });
    if (it == class_names_.end())
    {
      continue;
    }

    // reject class if confidence lower than threshold
    if (best_score < conf_threshold)
      continue;

    cv::Rect2f r(center_x - box_width * 0.5f, center_y - box_height * 0.5f, box_width, box_height);
    r = unletterbox(r);

    int left = std::max(0, (int)std::round(r.x));                            // X coordinate of the left edge
    int top = std::max(0, (int)std::round(r.y));                             // Y coordinate of the top edge
    int right = std::min(img_width - 1, (int)std::round(r.x + r.width));     // X coordinate of the right edge
    int bottom = std::min(img_height - 1, (int)std::round(r.y + r.height));  // Y coordinate of the bottom edge

    // re-calculate the height and width of the bounding box after clamping
    int new_width = std::max(0, right - left);   // Logic: right edge X xoordinate - Left edge X coordinate
    int new_height = std::max(0, bottom - top);  // Logic: right edge Y xoordinate - Left edge Y coordinate

    boxes.emplace_back(left, top, new_width, new_height);
    confidence_scores.emplace_back(best_score);
    class_ids.emplace_back(best_class);
  }
}

// Inspired from this: https://github.com/ultralytics/yolov5/issues/8427#issuecomment-1172469631
cv::Mat Yolo11DNNInference::letterbox(const cv::Mat& img, int newW, int newH, cv::Vec4i& pad, float& scale)
{
  int w = img.cols, h = img.rows;
  float r = std::min((float)newW / w, (float)newH / h);
  int nw = int(std::round(w * r)), nh = int(std::round(h * r));
  scale = r;
  int left = (newW - nw) / 2, top = (newH - nh) / 2;

  cv::Mat resized;
  cv::resize(img, resized, cv::Size(nw, nh));
  cv::Mat out(newH, newW, img.type(), cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(left, top, nw, nh)));
  pad = cv::Vec4i(left, top, newW - nw - left, newH - nh - top);
  return out;
}

}  // namespace yolo11_inference