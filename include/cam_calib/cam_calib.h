#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

namespace cam_calib {

class CamCalib {
public:
  CamCalib() : goodInput_(false), mode_(CAPTURING),
    chessBoardFlags_(cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK) {}

  CamCalib(ros::NodeHandle* nh);

  ~CamCalib() {}

private:
  enum InputType {INVALID, CAMERA, ROS_IMAGE_SUB};
  enum Mode {CAPTURING, CALIBRATED, UNDISTORTING};

  void imageCallback(const sensor_msgs::ImageConstPtr& msg);
  bool runCalibration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, cv::Size imageSize,
                      std::vector< std::vector<cv::Point2f> > imagePoints);
  void calcBoardCornerPositions(std::vector<cv::Point3f>& corners);
  double computeReprojectionErrors(const std::vector< std::vector<cv::Point3f> >& objectPoints,
                                   const std::vector< std::vector<cv::Point2f> >& imagePoints,
                                   const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                   const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                                   std::vector<float>& perViewErrors, bool fisheye);
  bool readXmlParam(const std::string& inputSettingsFile);
  void validate();
  void saveCameraParams(const cv::Size& imageSize, const cv::Mat& cameraMatrix,
                        const cv::Mat& distCoeffs, const std::string& outputFileName);

  cv::Size boardSize_;              // The size of the board -> Number of items by width and height
  float squareSize_;            // The size of a square in your defined unit (point, millimeter,etc).
  int nrFrames_;                // The number of frames to use from the input for calibration
  float aspectRatio_;           // The aspect ratio
  int delay_;                   // In case of a video input
  bool writePoints_;            // Write detected feature points
  bool writeExtrinsics_;        // Write extrinsic parameters
  bool calibZeroTangentDist_;   // Assume zero tangential distortion
  bool calibFixPrincipalPoint_; // Fix the principal point at the center
  bool flipVertical_;           // Flip the captured images around the horizontal axis
  bool showUndistorsed_;        // Show undistorted images after calibration
  std::string outputFileName_;       // The name of the file where to write
  std::string input_;                // The input ->
  bool useFisheye_;             // use fisheye camera model for calibration
  bool fixK1_;                  // fix K1 distortion coefficient
  bool fixK2_;                  // fix K2 distortion coefficient
  bool fixK3_;                  // fix K3 distortion coefficient
  bool fixK4_;                  // fix K4 distortion coefficient
  bool fixK5_;                  // fix K5 distortion coefficient

  int cameraID_;
  std::vector<std::string> imageList_;
  std::size_t atImageList_;
  cv::VideoCapture inputCapture_;
  InputType inputType_;
  bool goodInput_;
  int flag_;

  image_transport::Subscriber imageSub_;
  cv::Mat imageCurr_;
  std::string inputTopic_;
  Mode mode_;
  int chessBoardFlags_;
  std::vector< std::vector<cv::Point2f> > imagePoints_;
  cv::Mat camMat_;
  cv::Mat newCamMat_;
  cv::Mat distCoeffs_;
  cv::Mat map1_, map2_;
};

}
























