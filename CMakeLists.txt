cmake_minimum_required(VERSION 2.8.3)
project(cam_calib)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  std_msgs
  image_transport
)

find_package(OpenCV REQUIRED) # dhl
add_definitions(-std=c++11)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
)

add_executable(cam_calib_node src/cam_calib_node.cpp)
target_link_libraries(cam_calib_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
