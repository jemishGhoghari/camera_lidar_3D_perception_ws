#ifndef YOLO11_DNN_INFERENCE_HPP_
#define YOLO11_DNN_INFERENCE_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cvconfig.h>
#include <rclcpp/rclcpp.hpp>

#include <unordered_set>

/**
 * @file yolo_dnn_inference.hpp
 * @brief Header file for YOLOv11 DNN inference implementation.
 *
 * This file contains the definition of the Yolo11DNNInference class and related structures
 * for performing object detection using a YOLOv11 deep neural network. It includes configuration
 * options, utility functions, and methods for inference and post-processing.
 */

namespace yolo11_inference
{

/**
 * @struct YoloDNNConfig
 * @brief Configuration structure for YOLOv11 DNN inference.
 *
 * This structure holds the configuration parameters required for setting up the YOLOv11
 * deep neural network.
 *
 * @var modelPath
 * Path to the YOLOv11 model file.
 * @var confThreshold
 * Confidence threshold for filtering weak detections. Default is 0.25.
 * @var nmsThreshold
 * Non-maximum suppression (NMS) threshold for filtering overlapping boxes. Default is 0.45.
 * @var networkInputWidth
 * Width of the network input image. Default is 640.
 * @var networkInputHeight
 * Height of the network input image. Default is 640.
 * @var use_cuda
 * Flag to enable CUDA for inference. Default is false.
 */
struct YoloDNNConfig
{
  std::string modelPath;
  float confThreshold = 0.25f;
  float nmsThreshold = 0.45f;
  int networkInputWidth = 640;
  int networkInputHeight = 640;
  bool use_cuda = false;
};

/**
 * @struct DetectionResult
 * @brief Structure to store the results of object detection.
 *
 * This structure holds the bounding boxes, confidence scores, and class IDs
 * of detected objects.
 *
 * @var boxes
 * Vector of bounding boxes for detected objects.
 * @var confidence_scores
 * Vector of confidence scores for each detected object.
 * @var class_ids
 * Vector of class IDs corresponding to detected objects.
 */
struct DetectionResult
{
  std::vector<cv::Rect> boxes;
  std::vector<float> confidence_scores;
  std::vector<int> class_ids;
};

/**
 * @brief Loads the COCO class names.
 *
 * This function returns a vector of class names from the COCO dataset.
 *
 * @return A vector of strings containing COCO class names.
 */
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

/**
 * @brief Converts a 3D output matrix to a 2D matrix with dimensions (N x C).
 *
 * This utility function reshapes a 3D matrix to a 2D matrix, ensuring compatibility
 * with YOLO output formats.
 *
 * @param out3d The input 3D matrix.
 * @return A 2D matrix with dimensions (N x C).
 */
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

/**
 * @class Yolo11DNNInference
 * @brief Class for performing YOLOv11 DNN inference.
 *
 * This class encapsulates the functionality for loading a YOLOv11 model, performing
 * inference on input images, and decoding the detection results.
 */
class Yolo11DNNInference
{
public:
  /**
   * @brief Constructor for Yolo11DNNInference.
   *
   * Initializes the YOLOv11 DNN inference object with the given configuration,
   * class names, and logger.
   *
   * @param config Configuration parameters for YOLOv11 inference.
   * @param class_names Vector of class names for the model.
   * @param logger Logger for logging messages.
   */
  Yolo11DNNInference(const YoloDNNConfig& config, std::vector<std::string> class_names, const rclcpp::Logger& logger);

  /**
   * @brief Destructor for Yolo11DNNInference.
   */
  ~Yolo11DNNInference();

  /**
   * @brief Performs inference on an input image.
   *
   * This method processes the input image using the YOLOv11 model and populates
   * the detection results.
   *
   * @param input_image The input image for inference.
   * @param result The structure to store detection results.
   * @return True if inference is successful, false otherwise.
   */
  bool infer(const cv::Mat& input_image, DetectionResult& result);

  /**
   * @brief Retrieves the class names used by the model.
   *
   * @return A constant reference to the vector of class names.
   */
  const std::vector<std::string>& classNames() const;

private:
  /**
   * @brief Decodes the detection results from the model output.
   *
   * This method processes the raw output from the YOLOv11 model and extracts
   * bounding boxes, confidence scores, and class IDs.
   *
   * @param output The raw output from the model.
   * @param conf_threshold Confidence threshold for filtering detections.
   * @param K Number of top detections to consider.
   * @param img_width Width of the input image.
   * @param img_height Height of the input image.
   * @param pad Padding applied during preprocessing.
   * @param scale Scaling factor applied during preprocessing.
   * @param boxes Vector to store bounding boxes.
   * @param confidence_scores Vector to store confidence scores.
   * @param class_id Vector to store class IDs.
   */
  void decodeDetections(const cv::Mat& output, float conf_threshold, int K, int img_width, int img_height,
                        const cv::Vec4i& pad, float scale, std::vector<cv::Rect>& boxes,
                        std::vector<float>& confidence_scores, std::vector<int>& class_id);

  /**
   * @brief Preprocesses the input image using letterbox resizing.
   *
   * This method resizes the input image to the desired dimensions while maintaining
   * aspect ratio, adding padding as needed.
   *
   * @param img The input image.
   * @param newW The desired width of the resized image.
   * @param newH The desired height of the resized image.
   * @param pad Vector to store padding values.
   * @param scale Scaling factor applied during resizing.
   * @return The preprocessed image.
   */
  cv::Mat letterbox(const cv::Mat& img, int newW, int newH, cv::Vec4i& pad, float& scale);

  cv::dnn::Net net_;                       ///< The YOLOv11 DNN model.
  YoloDNNConfig config_;                   ///< Configuration parameters for the model.
  rclcpp::Logger logger_;                  ///< Logger for logging messages.
  std::vector<std::string> class_names_;   ///< Class names used by the model.
  std::vector<std::string> coco_classes_;  ///< COCO class names.
};

}  // namespace yolo11_inference

#endif  // YOLO11_DNN_INFERENCE_HPP_