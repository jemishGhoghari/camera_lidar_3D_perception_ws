#ifndef YOLO11_DNN_INFERENCE_HPP_
#define YOLO11_DNN_INFERENCE_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cvconfig.h>
#include <rclcpp/rclcpp.hpp>

#include <unordered_set>

namespace yolo11_inference
{

struct YoloDNNConfig
{
  std::string modelPath;
  float confThreshold = 0.25f;
  float nmsThreshold = 0.45f;
  int networkInputWidth = 640;
  int networkInputHeight = 640;
  bool use_cuda = false;
};

struct DetectionResult
{
  std::vector<cv::Rect> boxes;
  std::vector<float> confidence_scores;
  std::vector<int> class_ids;
};

static std::vector<std::string> loadCOCOClasses()
{
  return std::vector<std::string>{
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
  };
}

static cv::Mat toNxC(const cv::Mat& out3d)
{
  CV_Assert(out3d.dims >= 2);
  if (out3d.dims == 2)
  {
    // Already (N x C)
    return out3d;
  }
  CV_Assert(out3d.dims == 3 && out3d.size[0] == 1);
  const int d1 = out3d.size[1];
  const int d2 = out3d.size[2];

  const bool d1_is_C = (d1 <= 512);
  const bool d2_is_C = (d2 <= 512);

  if (d1_is_C && !d2_is_C)
  {
    // (1, C, N) -> (C x N) -> (N x C)
    cv::Mat cxn = out3d.reshape(1, d1);
    return cxn.t();
  }
  else if (!d1_is_C && d2_is_C)
  {
    // (1, N, C) -> (N x C)
    return out3d.reshape(1, d1);
  }
  else
  {
    // Ambiguous -> pick the smaller as C
    if (d1 <= d2)
    {
      cv::Mat cxn = out3d.reshape(1, d1);
      return cxn.t();
    }
    else
    {
      return out3d.reshape(1, d1);
    }
  }
}

class Yolo11DNNInference
{
public:
  Yolo11DNNInference(const YoloDNNConfig& config, std::vector<std::string> class_names, const rclcpp::Logger& logger);
  ~Yolo11DNNInference();

  bool infer(const cv::Mat& input_image, DetectionResult& result);

  const std::vector<std::string>& classNames() const
  {
    return class_names_;
  }

private:
  void decodeDetections(const cv::Mat& output, float conf_threshold, int K, int img_width, int img_height,
                        const cv::Vec4i& pad, float scale, std::vector<cv::Rect>& boxes,
                        std::vector<float>& confidence_scores, std::vector<int>& class_id);

  cv::Mat letterbox(const cv::Mat& img, int newW, int newH, cv::Vec4i& pad, float& scale);

  cv::dnn::Net net_;
  YoloDNNConfig config_;
  rclcpp::Logger logger_;
  std::vector<std::string> class_names_;
  std::vector<std::string> coco_classes_;
};

}  // namespace yolo11_inference

#endif  // YOLO11_DNN_INFERENCE_HPP_