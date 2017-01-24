#include <cam_calib/cam_calib.h>

namespace cam_calib {

CamCalib::CamCalib(ros::NodeHandle* nh) {
  CamCalib();
  const std::string inputSettingsFile("/home/dh/qt_catkin_ws/src/cam_calib/param/cal_set_param.xml");
  if (!readXmlParam(inputSettingsFile))
    return;

  if (inputType_ == ROS_IMAGE_SUB) {
    std::cout << "Subscribe ROS image\n";
    image_transport::ImageTransport imageTransp(*nh);
    imageSub_ = imageTransp.subscribe(inputTopic_, 1, &CamCalib::imageCallback, this);
  }
}


void CamCalib::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  imageCurr_ = cv_bridge::toCvShare(msg, msg->encoding)->image;
  std::vector<cv::Point2f> pointBuf;
  if (mode_ == CAPTURING) {
    bool found = cv::findChessboardCorners(imageCurr_, boardSize_, pointBuf, chessBoardFlags_);
    if (found) {
      cv::drawChessboardCorners(imageCurr_, boardSize_, cv::Mat(pointBuf), found);
      imagePoints_.push_back(pointBuf);
      if (imagePoints_.size() >= (size_t)nrFrames_) {
        if (runCalibration(camMat_, distCoeffs_, imageCurr_.size(), imagePoints_)) {
          mode_ = CALIBRATED;
        } else {
          ROS_ERROR("Calibration failed! Restart!");
          imagePoints_.clear();
        }
      }
    } else {
      std::cout << "Cannot find corners\n";
    }
    std::string showText = cv::format("%d / %d", (int)imagePoints_.size(), nrFrames_);
    cv::putText(imageCurr_, showText, cv::Point(60, 60), 1, 3, cv::Scalar(0, 255, 0));
  } else if (mode_ == CALIBRATED) {
    std::cout << "Done calibration!\n";

    saveCameraParams(imageCurr_.size(), camMat_, distCoeffs_, outputFileName_);

    if (useFisheye_) {
      cv::fisheye::estimateNewCameraMatrixForUndistortRectify(camMat_, distCoeffs_, imageCurr_.size(),
                                                              cv::Matx33d::eye(), newCamMat_, 1);
      cv::fisheye::initUndistortRectifyMap(camMat_, distCoeffs_, cv::Matx33d::eye(), newCamMat_,
                                           imageCurr_.size(), CV_16SC2, map1_, map2_);
    } else {
      cv::initUndistortRectifyMap(camMat_, distCoeffs_, cv::Mat(),
                                  cv::getOptimalNewCameraMatrix(camMat_, distCoeffs_, imageCurr_.size(), 1,
                                                                imageCurr_.size(), 0),
                                  imageCurr_.size(), CV_16SC2, map1_, map2_);
    }
    mode_ = UNDISTORTING;


  } else if (mode_ == UNDISTORTING) {
    cv::Mat temp = imageCurr_.clone();
    cv::remap(temp, imageCurr_, map1_, map2_, cv::INTER_LINEAR);
  }
  cv::imshow("Chessboard Detection View", imageCurr_);
  cv::waitKey(delay_);
}

bool CamCalib::runCalibration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, cv::Size imageSize,
                    std::vector< std::vector<cv::Point2f> > imagePoints) {

  std::vector<cv::Mat> rvecs, tvecs;
  std::vector<float> reprojErrs;
  double totalAvgErr = 0;

  //! [fixed_aspect]
  cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
  if (flag_ & cv::CALIB_FIX_ASPECT_RATIO )
    cameraMatrix.at<double>(0,0) = aspectRatio_;

  //! [fixed_aspect]
  if (useFisheye_)
    distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
  else
    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

  std::vector< std::vector<cv::Point3f> > objectPoints(1);
  calcBoardCornerPositions(objectPoints[0]);
  objectPoints.resize(imagePoints.size(),objectPoints[0]);

  //Find intrinsic and extrinsic camera parameters
  double rms;

  if (useFisheye_) {
    cv::Mat _rvecs, _tvecs;
    rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
                                 _rvecs, _tvecs, flag_);
    rvecs.reserve(_rvecs.rows);
    tvecs.reserve(_tvecs.rows);
    for(int i = 0; i < int(objectPoints.size()); i++){
        rvecs.push_back(_rvecs.row(i));
        tvecs.push_back(_tvecs.row(i));
    }
  } else {
    rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
                          rvecs, tvecs, flag_);
  }

  std::cout << "Re-projection error reported by calibrateCamera: "<< rms << std::endl;
  bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);
  totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                          distCoeffs, reprojErrs, useFisheye_);
  std::cout << (ok ? "Calibration succeeded" : "Calibration failed")
            << ". avg re projection error = " << totalAvgErr << std::endl;
  return ok;
}

void CamCalib::calcBoardCornerPositions(std::vector<cv::Point3f>& corners) {
  corners.clear();
  for (int i = 0; i < boardSize_.height; ++i) {
    for (int j = 0; j < boardSize_.width; ++j) {
      corners.push_back(cv::Point3f(j*squareSize_, i*squareSize_, 0));
    }
  }
}

double CamCalib::computeReprojectionErrors(const std::vector< std::vector<cv::Point3f> >& objectPoints,
                                 const std::vector< std::vector<cv::Point2f> >& imagePoints,
                                 const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                 const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                                 std::vector<float>& perViewErrors, bool fisheye) {
  std::vector<cv::Point2f> imagePoints2;
  std::size_t totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());

  for (std::size_t i = 0; i < objectPoints.size(); ++i ) {
    if (fisheye) {
      cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix, distCoeffs);
    } else {
      cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
    }
    err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

    std::size_t n = objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err*err/n);
    totalErr        += err*err;
    totalPoints     += n;
  }
  return std::sqrt(totalErr/totalPoints);
}

bool CamCalib::readXmlParam(const std::string& inputSettingsFile) {
  cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings
  if (!fs.isOpened()) {
    std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl;
    return false;
  }

  cv::FileNode node = fs.getFirstTopLevelNode();
  if (node.empty()) {
    std::cout << "Fail reading\n";
    fs.release();
    return false;
  } else {
    node["BoardSize_Width" ] >> boardSize_.width;
    node["BoardSize_Height"] >> boardSize_.height;
    node["Square_Size"]  >> squareSize_;
    node["Calibrate_NrOfFrameToUse"] >> nrFrames_;
    node["Calibrate_FixAspectRatio"] >> aspectRatio_;
    node["Write_DetectedFeaturePoints"] >> writePoints_;
    node["Write_extrinsicParameters"] >> writeExtrinsics_;
    node["Write_outputFileName"] >> outputFileName_;
    node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist_;
    node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint_;
    node["Calibrate_UseFisheyeModel"] >> useFisheye_;
    node["Input_FlipAroundHorizontalAxis"] >> flipVertical_;
    node["Show_UndistortedImage"] >> showUndistorsed_;
    node["Input"] >> input_;
    node["Input_topic"] >> inputTopic_;
    node["Input_Delay"] >> delay_;
    node["Fix_K1"] >> fixK1_;
    node["Fix_K2"] >> fixK2_;
    node["Fix_K3"] >> fixK3_;
    node["Fix_K4"] >> fixK4_;
    node["Fix_K5"] >> fixK5_;
    validate();
  }

  fs.release();
  if (!goodInput_) {
    std::cout << "Invalid input detected. Application stopping. " << std::endl;
    return false;
  }
  std::cout << "Succeed in receiving parameters. " << std::endl;
  return true;
}


void CamCalib::validate() {
  goodInput_ = true;
  if (boardSize_.width <= 0 || boardSize_.height <= 0) {
    std::cerr << "Invalid Board size: " << boardSize_.width << " " << boardSize_.height << std::endl;
    goodInput_ = false;
  }
  if (squareSize_ <= 10e-6) {
    std::cerr << "Invalid square size " << squareSize_ << std::endl;
    goodInput_ = false;
  }
  if (nrFrames_ <= 0) {
    std::cerr << "Invalid number of frames " << nrFrames_ << std::endl;
    goodInput_ = false;
  }

  inputType_ = INVALID;
  if (!input_.empty()) {      // Check for valid input
    if (input_[0] >= '0' && input_[0] <= '9') { // Image from cameras
      std::stringstream ss(input_);
      ss >> cameraID_;
      inputType_ = CAMERA;
      inputCapture_.open(cameraID_);
    } else if (input_.compare("ROS_image") == 0) { // Temporary
      std::cout << "ROS camera!!!\n";
      inputType_ = ROS_IMAGE_SUB;
    }
  }

  if (inputType_ == INVALID) {
    std::cerr << " Input does not exist: " << input_;
    goodInput_ = false;
  }

  flag_ = 0;
  if (calibFixPrincipalPoint_) flag_ |= cv::CALIB_FIX_PRINCIPAL_POINT;
  if (calibZeroTangentDist_)   flag_ |= cv::CALIB_ZERO_TANGENT_DIST;
  if (aspectRatio_)            flag_ |= cv::CALIB_FIX_ASPECT_RATIO;
  if (fixK1_)                  flag_ |= cv::CALIB_FIX_K1;
  if (fixK2_)                  flag_ |= cv::CALIB_FIX_K2;
  if (fixK3_)                  flag_ |= cv::CALIB_FIX_K3;
  if (fixK4_)                  flag_ |= cv::CALIB_FIX_K4;
  if (fixK5_)                  flag_ |= cv::CALIB_FIX_K5;

  if (useFisheye_) {
    // the fisheye model has its own enum, so overwrite the flags
    flag_ = cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    if (fixK1_)                flag_ |= cv::fisheye::CALIB_FIX_K1;
    if (fixK2_)                flag_ |= cv::fisheye::CALIB_FIX_K2;
    if (fixK3_)                flag_ |= cv::fisheye::CALIB_FIX_K3;
    if (fixK4_)                flag_ |= cv::fisheye::CALIB_FIX_K4;
  }
  atImageList_ = 0;
}

void CamCalib::saveCameraParams(const cv::Size& imageSize, const cv::Mat& cameraMatrix,
                      const cv::Mat& distCoeffs, const std::string& outputFileName) {
  cv::FileStorage fs(outputFileName, cv::FileStorage::WRITE);
  fs << "image_width" << imageSize.width;
  fs << "image_height" << imageSize.height;
  fs << "camera_matrix" << cameraMatrix;
  fs << "distortion_coefficients" << distCoeffs;
}

}


int main(int argc, char **argv) {
  ros::init(argc, argv, "cam_calib");
  ros::NodeHandle nh;
  cam_calib::CamCalib fcal(&nh);
  ros::spin();
  return 0;
}






























