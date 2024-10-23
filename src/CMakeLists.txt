cmake_minimum_required(VERSION 2.8.3)
project(livox_cam_calib)
add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  rospy
  std_msgs
  vision_msgs
  dynamic_reconfigure
)
generate_dynamic_reconfigure_options(
  config/bounds.cfg
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES colored_pointcloud
#  CATKIN_DEPENDS cv_bridge image_transport message_filters roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
#Eigen 
find_package(Eigen3)
include_directories( ${EIGEN3_INCLUDE_DIR} )

# opencv
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pcl
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

#Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Sophus
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
set(Sophus_LIBRARIES /usr/local/lib/libSophus.so)
## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(livox_cam_calib src/livox_cam_calib.cpp src/LidarCornersEst.cpp src/Optimization.cpp src/Visualization.cpp src/ImageCornersEst.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(livox_cam_calib ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${CERES_LIBRARIES} ${Sophus_LIBRARIES})



#########install##############
 install(TARGETS  livox_cam_calib 
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

## 添加头文件．Mark cpp header files for installation
 install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
 )

 install(DIRECTORY launch
           DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
 install(DIRECTORY rviz
           DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
 install(DIRECTORY config
           DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
  install(FILES
   nodelet_plugins.xml
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
 )

