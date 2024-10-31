/*
 * @Author: huangwei barry.huangw@gmail.com
 * @Date: 2024-09-13 15:22:51
 * @LastEditors: huangwei barry.huangw@gmail.com
 * @LastEditTime: 2024-10-22 13:44:48
 * @FilePath: /livox_cam_calib/src/src/livox_cam_calib.cpp
 * @Description: Lidar和Camera的联合自动标定程序
 */

#include "ImageCornersEst.h"
#include <Eigen/Dense>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <math.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <termios.h> // For terminal input
#include <unistd.h>  // For read()

#include "common.hpp"
#include "types.h"
#include <boost/thread.hpp>
#include <cstdio>
#include <ctime>
#include <dynamic_reconfigure/server.h>
#include <livox_cam_calib/boundsConfig.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <vision_msgs/Detection2DArray.h>

#include "LidarCornersEst.h"
#include "pcl_ros/transforms.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/intersections.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/impl/extract_indices.hpp>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/filters/impl/project_inliers.hpp>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/MarkerArray.h>
#define YELLOW "\033[33m" /* Yellow */
#define GREEN "\033[32m"  /* Green */
#define REND "\033[0m" << std::endl

#define WARN (std::cout << YELLOW)
#define INFO (std::cout << GREEN)

using namespace common;

ros::Publisher image_corners_pub, original_cloud_pub, corner_pub, crop_pub,
    pca_pub;

class RsCamFusion {
public:
  RsCamFusion(cv::Mat matrix, cv::Size corners_num, double grid_size,
              std::string res_filename, std::string cam_yaml, int save_num) {
    intrinsic_ = matrix;
    corner_size_ = corners_num;
    grid_size_ = grid_size;
    res_save_file_name_ = res_filename;
    calib_num_ = save_num;
    lidar_corners_est_.reset(new LidarCornersEst());
    lidar_corners_est_->set_chessboard_param(corner_size_, grid_size_);
    image_corners_est_.reset(new ImageCornersEst(cam_yaml));
    common_ptr_.reset(new Common(corner_size_));

    boost::thread keyboard_thread(&RsCamFusion::keyboardListener, this);
  }

  void keyboardListener() {
    char c;
    while (ros::ok()) {
      c = getch();    // 获取按键输入
      if (c == 'n') { // 按下'n'键时，允许执行下一次LiDAR回调
        trigger_lidar_callback_ = true;
        trigger_camera_callback_ = true;
        std::cout
            << "Key 'n' pressed. Ready for next lidar and camera callback."
            << std::endl;
      } else if (c == 'k') { // 按下'k' 保存参数
        trigger_params_save_ = true;
        save_info();
        calcExtrinixMatrix();
        std::cout << "Current frame is good, save the corners." << std::endl;
      } else if (c == 'c') {
        std::cout << "Calculation of Mtraix." << std::endl;
      }
    }
  }

  // 非阻塞获取键盘输入的函数
  int getch() {
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
  }

  // 回调函数，当用户在 RViz 中点击点时调用
  void pointCallback(const geometry_msgs::PointStamped::ConstPtr &msg) {
    selected_point_ = *msg; // 保存选中的点
    ROS_INFO("Selected point: x = %f, y = %f, z = %f", selected_point_.point.x,
             selected_point_.point.y, selected_point_.point.z);

    // 发布空的点云来清除RViz中的历史点云
    VPointCloud::Ptr empty_cloud1(new VPointCloud); // 创建一个空的点云
    VPointCloud::Ptr empty_cloud2(new VPointCloud); // 创建一个空的点云
    common_ptr_->publish_pointcloud(crop_pub, lidar_header_,
                                    empty_cloud1); // 发布空点云
    common_ptr_->publish_pointcloud(corner_pub, lidar_header_,
                                    empty_cloud2); // 发布空点云

    VPointCloud::Ptr cropped_cloud(new VPointCloud);
    // 切割标定板的点云并发布
    *cropped_cloud = *input_cloud_ptr_;
    lidar_corners_est_->CropBox(selected_point_, cropped_cloud);
    common_ptr_->publishCloudtoShow(crop_pub, lidar_header_, cropped_cloud);
    // 对标定板点云进行平面拟合和角点计算，并发布出去

    lidar_corner_points_.reset(new VPointCloud);
    if (!lidar_corners_est_->detectCornersFromLidar(cropped_cloud,
                                                    lidar_corner_points_)) {
      ROS_WARN("Lidar Corners detection fails");
      return;
    }
    common_ptr_->publishCloudtoShow(corner_pub, lidar_header_,
                                    lidar_corner_points_);
  }

  void camera_callback(const sensor_msgs::ImageConstPtr input_image_msg) {
    if (!trigger_camera_callback_) {
      return; // 如果条件不满足，跳过回调执行
    }
    trigger_camera_callback_ = false;
    cv::Mat undistorted_image;
    cv_bridge::CvImagePtr cv_ptr;

    try {
      cv_ptr = cv_bridge::toCvCopy(input_image_msg,
                                   sensor_msgs::image_encodings::TYPE_8UC3);
    } catch (cv_bridge::Exception e) {
      ROS_ERROR_STREAM("Cv_bridge Exception:" << e.what());
      return;
    }
    input_image_ = cv_ptr->image;

    image_corners_.clear();
    // opencv的角点检测并发布出去，方便进行检查
    if (!image_corners_est_->detectCornersFromCam(
            input_image_.clone(), image_corners_, image_with_corners_)) {
      cam_in = false;
      return;
    }
    common_ptr_->publishImage(image_corners_pub, lidar_header_,
                              image_with_corners_);

    cam_in = true;
  }

  void lidar_callback(const sensor_msgs::PointCloud2ConstPtr input_cloud_msg) {
    if (!cam_in || !trigger_lidar_callback_) {
      return; // 如果条件不满足，跳过回调执行
    }
    // 允许执行LiDAR回调后，立即将标志位重置
    trigger_lidar_callback_ = false;
    // 发布空的点云来清除RViz中的历史点云
    VPointCloud::Ptr empty_cloud(new VPointCloud); // 创建一个空的点云
    common_ptr_->publish_pointcloud(original_cloud_pub, lidar_header_,
                                    empty_cloud); // 发布空点云
    lidar_header_ = input_cloud_msg->header;
    input_cloud_ptr_.reset(new VPointCloud);
    pcl::fromROSMsg(*input_cloud_msg, *input_cloud_ptr_);
    common_ptr_->publish_pointcloud(original_cloud_pub, lidar_header_,
                                    input_cloud_ptr_);
  }

private: // 按键监听函数，设置一个专门的线程来处理按键输入
  void calcExtrinixMatrix() {
    point3d_.clear();
    point2d_.clear();
    // 读取文件txt的cam和lidar的角点信息
    for (auto &p : *lidar_corner_points_) {
      point3d_.push_back(Eigen::Vector3d(p.x, p.y, p.z));
    }
    point2d_.insert(point2d_.end(), image_corners_.begin(),
                    image_corners_.end());

    Eigen::Isometry3d T_lidar2cam_axis_roughly =
        common_ptr_->get_lidar2cam_axis_roughly();

    for (unsigned int i = 0; i < point3d_.size();
         i++) /// 这个是为了让相机和激光坐标系方向一致
      point3d_.at(i) = T_lidar2cam_axis_roughly * point3d_.at(i);

    image_corners_est_->check_order_lidar(point3d_, corner_size_);
    image_corners_est_->check_order_cam(point2d_, corner_size_);

    lidar_corners_piece_.push_back(point3d_);
    image_corners_piece_.push_back(point2d_);

    if (calib_num_ == 0) {
      ROS_WARN("R, T calculation！");
      // 数量等于预设的对数后 进行pnp算法匹配
      // data ready
      std::vector<Eigen::Vector3d> lidar_corners;
      std::vector<cv::Point2d> image_corners;

      for (unsigned int i = 0; i < lidar_corners_piece_.size(); i++) {
        lidar_corners.insert(lidar_corners.end(),
                             lidar_corners_piece_.at(i).begin(),
                             lidar_corners_piece_.at(i).end());
        image_corners.insert(image_corners.end(),
                             image_corners_piece_.at(i).begin(),
                             image_corners_piece_.at(i).end());
      }

      std::cout << lidar_corners.size() << std::endl;
      std::cout << image_corners.size() << std::endl;
      std::cout << m_images_.size() << std::endl;
      //完善内参参数＆畸变系数参数

      Eigen::Vector3d r_ceres(0, 0, 0);
      Eigen::Vector3d t_ceres(0, 0, 0);
      Optimization optim;
      Eigen::Isometry3d T_lidar2cam = optim.solvePose3d2dError(
          lidar_corners, image_corners, intrinsic_, r_ceres, t_ceres);

      Eigen::Matrix3d R = T_lidar2cam.rotation();
      Eigen::Vector3d t = T_lidar2cam.translation();
      image_corners_est_->setRt(R, t);

      // 初始化变换矩阵
      T_lidar2cam = T_lidar2cam * T_lidar2cam_axis_roughly;
      std::cout << "lidar2cam:\n" << T_lidar2cam.matrix() << std::endl;
      std::cout << "cam2lidar:\n"
                << T_lidar2cam.inverse().matrix() << std::endl;

      // 保存对应结果
      image_corners_est_->extrinsic2txt(
          res_save_file_name_ + "extrinsicParams.txt", T_lidar2cam.matrix());

      // // 验证变换结果
      for (unsigned int i = 0; i < m_images_.size(); i++) {
        cv::Mat image = m_images_.at(i);
        // cv::cvtColor(m_images.at(i), image, CV_GRAY2RGB);
        VPointCloud::Ptr cloud_ptr(new VPointCloud);
        VPointCloud::Ptr transformed_cloud_ptr(new VPointCloud);
        for (const auto &point_vec : lidar_corners_piece_.at(i)) {
          pcl::PointXYZI point;
          point.x = point_vec.x();
          point.y = point_vec.y();
          point.z = point_vec.z();

          // 将转换后的点加入点云对象
          cloud_ptr->points.push_back(point);
        }
        // 步骤 1: 获取 4x4 的 Matrix4d
        Eigen::Matrix4d matrix_4d = T_lidar2cam.matrix();

        // 步骤 2: 转换为 4x4 的 Matrix4f
        Eigen::Matrix4f matrix_4f = matrix_4d.cast<float>();

        pcl::transformPointCloud(*cloud_ptr, *transformed_cloud_ptr, matrix_4f);

        image_corners_est_->show_calib_result(
            lidar_corners_piece_.at(i), image_corners_piece_.at(i), image);

        cv::imwrite("/home/jyzn/res_tray_detection" + std::to_string(i) +
                        ".png",
                    image);
        cv::imshow(std::to_string(i + 1), image);
        cv::waitKey(0);
      }
    }
  }

  void save_info() {
    // save the corners info
    std::string lidar_save_path = res_save_file_name_ + "lidar_corners_" +
                                  std::to_string(calib_num_) + ".txt";
    std::string cam_save_path = res_save_file_name_ + "cam_corners_" +
                                std::to_string(calib_num_) + ".txt";
    std::string image_path =
        res_save_file_name_ + "pic_" + std::to_string(calib_num_) + ".jpg";
    std::string image_with_corners_path = res_save_file_name_ +
                                          "pic_with_corners_" +
                                          std::to_string(calib_num_) + ".jpg";
    std::string lidar_points_save_path = res_save_file_name_ + "lidar_points_" +
                                         std::to_string(calib_num_) + ".txt";
    std::string cam_points_save_path = res_save_file_name_ + "cam_points_" +
                                       std::to_string(calib_num_) + ".txt";
    // 图像保存
    calib_num_--;
    common_ptr_->save_corners2txt(lidar_corner_points_, lidar_save_path);
    common_ptr_->save_corners2txt(image_corners_, cam_save_path);
    for (auto &p : *lidar_corner_points_) {
      point3d_.push_back(Eigen::Vector3d(p.x, p.y, p.z));
    }

    point2d_.insert(point2d_.end(), image_corners_.begin(),
                    image_corners_.end());

    Eigen::Isometry3d T_lidar2cam_axis_roughly =
        common_ptr_->get_lidar2cam_axis_roughly();

    for (unsigned int i = 0; i < point3d_.size();
         i++) /// 这个是为了让相机和激光坐标系方向一致
      point3d_.at(i) = T_lidar2cam_axis_roughly * point3d_.at(i);

    image_corners_est_->check_order_lidar(point3d_, corner_size_);
    image_corners_est_->check_order_cam(point2d_, corner_size_);

    common_ptr_->savePoint3dToFile(lidar_points_save_path, point3d_);
    common_ptr_->savePoint2dToFile(cam_points_save_path, point2d_);
    cv::imwrite(image_with_corners_path, image_with_corners_.clone());
    cv::imwrite(image_path, input_image_.clone());
    m_images_.push_back(input_image_.clone());
  }

  std::string res_save_file_name_; // 数据保存路径
  int calib_num_;                  // 标定数据数量
  bool cam_in;
  std_msgs::Header lidar_header_;

  // ------------ Camera ------------//
  cv::Mat input_image_ = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat intrinsic_;
  cv::Size corner_size_;                   // 标定板角点数量
  double grid_size_;                       // 标定板的网格尺寸
  std::vector<cv::Point2f> image_corners_; // 图象角点信息
  cv::Mat image_with_corners_;             // 带有角点显示的图像
  std::vector<cv::Mat> m_images_;          // 图像列表

  ImageCornersEst::Ptr image_corners_est_; // 相机算法类

  // ------------ Lidar ------------- //
  VPointCloud::Ptr input_cloud_ptr_;           // 输入点云指针
  VPointCloud::Ptr lidar_corner_points_;       //激光角点信息
  LidarCornersEst::Ptr lidar_corners_est_;     // 雷达的角点检测算法类
  geometry_msgs::PointStamped selected_point_; // 选取的标定板中心点

  Common::Ptr common_ptr_; // 通用函数类

  // 用于结算R，T矩阵
  std::vector<Eigen::Vector3d> point3d_;
  std::vector<cv::Point2d> point2d_;

  std::vector<std::vector<Eigen::Vector3d>> lidar_corners_piece_;
  std::vector<std::vector<cv::Point2d>> image_corners_piece_;

  bool trigger_lidar_callback_ = false;  // Flag to control lidar callback
  bool trigger_camera_callback_ = false; // Flag to control lidar callback
  bool trigger_params_save_ = false;     // Flag to control params save
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "colored_pointcloud_node");
  ros::NodeHandle nh;
  ros::NodeHandle priv_nh("~");

  ros::Subscriber camera_sub, lidar_sub;

  std::string config_path, file_name, result_save_file_path;
  std::string camera_topic, lidar_topic;
  int bag_num; // 采集标定数据的数量
  cv::Size corner_size;
  double grid_size;
  cv::Mat CameraMatrix;

  if (priv_nh.hasParam("calib_file_path") && priv_nh.hasParam("file_name")) {
    priv_nh.getParam("camera_topic", camera_topic);
    priv_nh.getParam("lidar_topic", lidar_topic);
    priv_nh.getParam("calib_file_path", config_path);
    priv_nh.getParam("file_name", file_name);
    priv_nh.getParam("result_save_file_path", result_save_file_path);
    priv_nh.getParam("bag_num", bag_num);
  } else {
    WARN << "Config file is empty!" << REND;
    return 0;
  }

  INFO << "config path: " << config_path << REND;
  INFO << "config file: " << file_name << REND;
  INFO << "data save path : " << result_save_file_path << REND;

  std::string config_file_name = config_path + "/" + file_name;
  cv::FileStorage fs_reader(config_file_name, cv::FileStorage::READ);
  fs_reader["CameraMat"] >> CameraMatrix;
  fs_reader["ChessBoardSize"] >> corner_size; // 标定板的角点数量
  fs_reader["grid_length"] >> grid_size;      // 标定板网格的尺寸
  fs_reader.release();

  if (lidar_topic.empty() || camera_topic.empty()) {
    WARN << "sensor topic is empty!" << REND;
    return 0;
  }

  INFO << "image corners size: " << corner_size << REND;
  INFO << "image grid size: " << grid_size << REND;

  RsCamFusion fusion(CameraMatrix, corner_size, grid_size,
                     result_save_file_path, config_file_name, bag_num);

  // 订阅的回调函数
  camera_sub =
      nh.subscribe(camera_topic, 1, &RsCamFusion::camera_callback, &fusion);
  lidar_sub =
      nh.subscribe(lidar_topic, 1, &RsCamFusion::lidar_callback, &fusion);

  // 订阅 RViz 中的 Publish Point 工具发布的点
  ros::Subscriber point_sub =
      nh.subscribe("/clicked_point", 1, &RsCamFusion::pointCallback, &fusion);

  // 发布的节点
  original_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("src_cloud", 10);
  crop_pub = nh.advertise<sensor_msgs::PointCloud2>("crop_cloud", 10);
  pca_pub = nh.advertise<sensor_msgs::PointCloud2>("pca_cloud", 10);
  image_corners_pub = nh.advertise<sensor_msgs::Image>("image_corners", 10);
  corner_pub = nh.advertise<sensor_msgs::PointCloud2>("corners_cloud", 10);

  ros::spin();
  return 0;
}



#include "ImageCornersEst.h"
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp> /// cv::findChessboardCorners
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> /// cv::undistort

ImageCornersEst::ImageCornersEst(std::string cam_yaml) {

  cv::FileStorage fs;
  fs.open(cam_yaml, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    std::cerr << "can not open " << cam_yaml << std::endl;
    return;
  }
  fs["CameraMat"] >> camK;
  fs["DistCoeff"] >> distort_param;
  fs["ImageSize"]["width"] >> m_image_size.width;
  fs["ImageSize"]["height"] >> m_image_size.height;
  m_grid_length = static_cast<double>(fs["grid_length"]);
  fs["ChessBoardSize"] >> m_board_size;
  std::cout << "image size:" << m_image_size.width << "," << m_image_size.height
            << std::endl;

  fs.release();

  m_fx = camK.at<double>(0, 0);
  m_cx = camK.at<double>(0, 2);
  m_fy = camK.at<double>(1, 1);
  m_cy = camK.at<double>(1, 2);
  std::cout << "m_fx" << m_fx << "m_cx" << m_cx << "m_fy" << m_fy << "m_cy"
            << m_cy << std::endl;

  std::cout << "camK: \n" << camK << std::endl;
  std::cout << "dist: \n" << distort_param << std::endl;
  std::cout << "grid_length: " << m_grid_length << std::endl;
  std::cout << "corner_in_x: " << m_board_size.width << std::endl;
  std::cout << "corner_in_y: " << m_board_size.height << std::endl;
}

bool ImageCornersEst::getRectifyParam(std::string cam_yaml) { return true; }

void ImageCornersEst::undistort_image(cv::Mat image, cv::Mat &rectify_image) {
  cv::undistort(image, rectify_image, camK, distort_param, camK);
}

bool ImageCornersEst::spaceToPlane(Eigen::Vector3d P_w, Eigen::Vector2d &P_cam,
                                   double dis) {
  Eigen::Vector3d P_c = m_R * P_w + m_t;
  //        Eigen::Vector3d P_c = P_w;
  if (P_c[2] < 0 || P_c[2] > dis) {
    return false;
  }
  // Transform to model plane
  double u = P_c[0] / P_c[2];
  double v = P_c[1] / P_c[2];

  P_cam(0) = m_fx * u + m_cx;
  P_cam(1) = m_fy * v + m_cy;

  if (P_cam(0) > 0 && P_cam(0) < m_image_size.width && P_cam(1) > 0 &&
      P_cam(1) < m_image_size.height) {
    return true;
  } else
    return false;
}

void ImageCornersEst::show_calib_result(std::vector<Eigen::Vector3d> point3d,
                                        std::vector<cv::Point2d> point2d,
                                        cv::Mat &image) {
  float errorSum = 0.0f;
  float errorMax = 0;
  for (unsigned int i = 0; i < point3d.size(); i++) {
    Eigen::Vector3d Pw = point3d.at(i);
    Eigen::Vector2d Pcam;
    if (spaceToPlane(Pw, Pcam)) {

      cv::Point2d pEst(Pcam[0], Pcam[1]);
      cv::Point2d pObs = point2d.at(i);

      cv::circle(image, pEst, 1, cv::Scalar(0, 0, 255), 2); // r
      cv::circle(image, pObs, 1, cv::Scalar(255, 0, 0), 2); // b

      float error = cv::norm(pObs - pEst);

      errorSum += error;
      if (error > errorMax) {
        errorMax = error;
      }
    }
  }

  std::ostringstream oss;
  oss << "Reprojection error: avg = " << errorSum / 35.0
      << "   max = " << errorMax;

  cv::putText(image, oss.str(), cv::Point(10, image.rows - 20),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 0.5,
              cv::LINE_AA);
  std::cout << "  " << oss.str() << std::endl;
}

void ImageCornersEst::split(std::string &s, std::string &delim,
                            std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (index - last > 0) {
    ret.push_back(s.substr(last, index - last));
  }
}

void ImageCornersEst::read_cam_corners(std::string filename, int num,
                                       std::vector<cv::Point2d> &point2d) {
  std::vector<std::vector<cv::Point2d>> Corners;

  std::ifstream infile;
  infile.open(filename);

  std::string s;
  int counter = 0;
  std::string dlim = " ";
  while (std::getline(infile, s)) {
    std::vector<std::string> slist;
    split(s, dlim, slist);
    std::vector<cv::Point2d> corner;
    for (uint i = 0; i < slist.size(); i++) {
      std::stringstream ss;
      ss << slist[i];
      cv::Point2d p;
      ss >> p.x;
      p.y = 0;
      corner.push_back(p);
      counter++;
    }
    Corners.push_back(corner);

    if (counter >= num)
      break;
  }

  counter = 0;
  while (std::getline(infile, s)) {
    std::vector<std::string> slist;
    split(s, dlim, slist);
    for (uint i = 0; i < slist.size(); i++) {
      std::stringstream ss;
      ss << slist[i];
      ss >> Corners[counter][i].y;
    }
    counter++;
  }

  infile.close();

  /// ok
  if (Corners.size() != m_board_size.height) {
    for (unsigned int w = 0; w < Corners.at(0).size(); w++)
      for (unsigned int h = 0; h < Corners.size(); h++)
        point2d.push_back(Corners[h][w]);

  } else {
    for (unsigned int h = 0; h < Corners.size(); h++)
      for (unsigned int w = 0; w < Corners.at(0).size(); w++)
        point2d.push_back(Corners[h][w]);
  }

  // std::cout << " camera corner size: " << point2d.size() << std::endl;
}

void ImageCornersEst::read_lidar_corners(
    std::string filename, int num, std::vector<Eigen::Vector3d> &point3d) {

  std::ifstream infile;

  int counter = 0;
  infile.open(filename.c_str());
  while (!infile.eof() && counter < num) {
    float x, y, z;
    infile >> x >> y >> z;
    point3d.push_back(Eigen::Vector3d(x, y, z));

    counter++;
  }
  infile.close();

  // std::cout << " lidar  corner size: " << point3d.size() << std::endl;
}

bool ImageCornersEst::detectCornersFromCam(
    cv::Mat input_image, std::vector<cv::Point2f> &image_corners,
    cv::Mat &output_image) {
  cv::Mat src_image;
  cv::cvtColor(input_image, src_image, cv::COLOR_BGR2GRAY);
  bool whether_found =
      cv::findChessboardCorners(src_image, m_board_size, image_corners);
  if (whether_found) {
    cv::cornerSubPix(
        src_image, image_corners, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                         0.01));
    // 绘制检测到的角点
    cv::drawChessboardCorners(src_image, m_board_size, cv::Mat(image_corners),
                              whether_found);

    output_image = src_image.clone();
    std::cout << "角点检测成功!" << std::endl;
    return true;
  }
  std::cout << "角点检测失败!" << std::endl;
  return false;
}

void ImageCornersEst::extrinsic2txt(std::string savefile,
                                    Eigen::Matrix4d lidar2cam) {
  // 打开文件，以文本模式写入
  std::ofstream outfile(savefile.c_str());

  // 遍历矩阵的每个元素，并按行写入文件
  if (outfile.is_open()) {
    for (int i = 0; i < lidar2cam.rows(); ++i) {
      for (int j = 0; j < lidar2cam.cols(); ++j) {
        outfile << lidar2cam(i, j); // 输出矩阵元素
        if (j < lidar2cam.cols() - 1) {
          outfile << " "; // 在行的元素之间加空格
        }
      }
      outfile << std::endl; // 每行结束后换行
    }
  }

  outfile.close(); // 关闭文件
}

void ImageCornersEst::txt2extrinsic(std::string filepath) {
  Eigen::Matrix4d lidar2cam;
  std::ifstream infile(filepath, std::ios_base::binary);
  infile.read((char *)&lidar2cam, sizeof(Eigen::Matrix4d));
  infile.close();
  std::cout << "lidar2cam\n" << lidar2cam << std::endl;

  Eigen::Isometry3d T_lidar2cam_temp = Eigen::Isometry3d::Identity();
  T_lidar2cam_temp.rotate(lidar2cam.block<3, 3>(0, 0));
  T_lidar2cam_temp.pretranslate(lidar2cam.block<3, 1>(0, 3));

  m_R = lidar2cam.block<3, 3>(0, 0);
  m_t = lidar2cam.block<3, 1>(0, 3);

  T_lidar2cam = T_lidar2cam_temp;

  std::cout << "cam2lidar: \n"
            << T_lidar2cam_temp.inverse().matrix() << std::endl;
}

void ImageCornersEst::HSVtoRGB(int h, int s, int v, unsigned char *r,
                               unsigned char *g, unsigned char *b) {
  // convert from HSV/HSB to RGB color
  // R,G,B from 0-255, H from 0-260, S,V from 0-100
  // ref http://colorizer.org/

  // The hue (H) of a color refers to which pure color it resembles
  // The saturation (S) of a color describes how white the color is
  // The value (V) of a color, also called its lightness, describes how dark the
  // color is

  int i;

  float RGB_min, RGB_max;
  RGB_max = v * 2.55f;
  RGB_min = RGB_max * (100 - s) / 100.0f;

  i = h / 60;
  int difs = h % 60; // factorial part of h

  // RGB adjustment amount by hue
  float RGB_Adj = (RGB_max - RGB_min) * difs / 60.0f;

  switch (i) {
  case 0:
    *r = RGB_max;
    *g = RGB_min + RGB_Adj;
    *b = RGB_min;
    break;
  case 1:
    *r = RGB_max - RGB_Adj;
    *g = RGB_max;
    *b = RGB_min;
    break;
  case 2:
    *r = RGB_min;
    *g = RGB_max;
    *b = RGB_min + RGB_Adj;
    break;
  case 3:
    *r = RGB_min;
    *g = RGB_max - RGB_Adj;
    *b = RGB_max;
    break;
  case 4:
    *r = RGB_min + RGB_Adj;
    *g = RGB_min;
    *b = RGB_max;
    break;
  default: // case 5:
    *r = RGB_max;
    *g = RGB_min;
    *b = RGB_max - RGB_Adj;
    break;
  }
}

void ImageCornersEst::check_order_cam(std::vector<cv::Point2d> &point2d,
                                      cv::Size boardSize) {
  /// check point2D
  if (point2d.at(0).y > point2d.at(boardSize.width + 1).y) {
    for (int h = 0; h < boardSize.height / 2; h++) {
      int front = boardSize.width * h;
      int end = boardSize.width * (boardSize.height - 1 - h);
      for (int w = 0; w < boardSize.width; w++) {
        cv::Point2d p = point2d.at(front + w);
        point2d.at(front + w) = point2d.at(end + w);
        point2d.at(end + w) = p;
      }
    }
  }
  if (point2d.at(0).x > point2d.at(1).x) {
    for (int h = 0; h < boardSize.height; h++) {
      int front = boardSize.width * h;
      for (int w = 0; w < boardSize.width / 2; w++) {
        cv::Point2d p = point2d.at(front + w);
        point2d.at(front + w) = point2d.at(front + boardSize.width - 1 - w);
        point2d.at(front + boardSize.width - 1 - w) = p;
      }
    }
  }
}

void ImageCornersEst::check_order_lidar(std::vector<Eigen::Vector3d> &point3d,
                                        cv::Size boardSize) {
  /// check point3D
  if (point3d.at(0)[1] > point3d.at(boardSize.width + 1)[1]) {
    for (int h = 0; h < boardSize.height / 2; h++) {
      int front = boardSize.width * h;
      int end = boardSize.width * (boardSize.height - 1 - h);
      for (int w = 0; w < boardSize.width; w++) {
        Eigen::Vector3d p = point3d.at(front + w);
        point3d.at(front + w) = point3d.at(end + w);
        point3d.at(end + w) = p;
      }
    }
  }
  if (point3d.at(0)[0] > point3d.at(1)[0]) {
    for (int h = 0; h < boardSize.height; h++) {
      int front = boardSize.width * h;
      for (int w = 0; w < boardSize.width / 2; w++) {
        Eigen::Vector3d p = point3d.at(front + w);
        point3d.at(front + w) = point3d.at(front + boardSize.width - 1 - w);
        point3d.at(front + boardSize.width - 1 - w) = p;
      }
    }
  }
}





#include "LidarCornersEst.h"

#include <pcl/filters/passthrough.h> /// pcl::PassThrough

/// EuclideanCluster
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h> /// pcl::transformPointCloud
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h> /// pcl::visualization::PCLVisualizer

#include "Optimization.h"
#include <boost/thread/thread.hpp>

void LidarCornersEst::set_chessboard_param(cv::Size grid_info,
                                           double grid_legnth) {
  m_grid_length = grid_legnth;
  m_grid_size(0) = grid_info.width + 1;
  m_grid_size(1) = grid_info.height + 1;

  if (m_grid_size(0) > m_grid_size(1)) {
    int temp = m_grid_size(0);
    m_grid_size(0) = m_grid_size(1);
    m_grid_size(1) = temp;
  }

  std::cout << "grid_length: " << m_grid_length << std::endl;
  std::cout << "grid_in_x: " << m_grid_size(0) << std::endl;
  std::cout << "grid_in_y: " << m_grid_size(1) << std::endl;

  return;
}

Eigen::Vector2d LidarCornersEst::calHist(std::vector<double> datas) {
  int HISTO_LENGTH = 100;
  std::vector<int> dataHist; //反射强度直方图统计
  dataHist.reserve(HISTO_LENGTH);

  for (int i = 0; i < HISTO_LENGTH; i++)
    dataHist.push_back(0);
  sort(datas.begin(), datas.end());

  double min = datas.front();
  double max = datas.back();
  const double factor = HISTO_LENGTH / (max - min);

  for (unsigned int i = 0; i < datas.size(); i++) {
    double sample = datas.at(i) - min;
    int bin = round(sample * factor); // 将sample分配到bin组
    dataHist[bin]++;
  }

  double sum = 0.0;
  for (unsigned int i = 0; i < datas.size(); i++)
    sum += datas.at(i);
  double mean = sum / datas.size(); //均值

  double low_intensity = -1;
  double high_intensity = -1;

  bool low_found = false;
  bool high_found = false;

  std::cout << "min " << min << ", max " << max << ", mean " << mean
            << std::endl;

  double bin_width = (max - min) / HISTO_LENGTH;
  std::cout << "bin_width: " << bin_width << std::endl;

  std::map<double, int> hist_map;
  for (unsigned int i = 0; i < dataHist.size(); i++)
    hist_map.insert(std::make_pair(dataHist.at(i), i));
  std::map<double, int>::reverse_iterator iter;
  iter = hist_map.rbegin();
  while (!low_found || !high_found) {
    int index = iter->second;
    double bin_edge = bin_width * double(index) + min;

    if (bin_edge > mean && !high_found) {
      high_found = true;
      high_intensity = bin_edge;
    }
    if (bin_edge < mean && !low_found) {
      low_found = true;
      low_intensity = bin_edge;
    }
    iter++;
  }

  //    std::cout << low_intensity << " " <<  high_intensity << std::endl;

  /// 画出直方图统计图
  // #if 0
  //     cv::Mat image2 = cv::Mat::zeros(600,600, CV_8UC3);
  //     for(int i=0;i<HISTO_LENGTH;i++){

  //         double height=dataHist[i];//计算高度
  //         //画出对应的高度图
  //         cv::rectangle(image2,cv::Point(i*2,600), cv::Point((i+1)*2 - 1,
  //         600-height), CV_RGB(255,255,255));
  //     }
  //     cv::imshow("hist",image2);
  //     cv::waitKey(5);
  // #endif

  return Eigen::Vector2d(low_intensity, high_intensity);
}

Eigen::Vector2d LidarCornersEst::get_gray_zone(VPointCloud::Ptr &cloud,
                                               double rate) {
  Eigen::Vector2d gray_zone;
  std::vector<Eigen::Vector3d> point3d;
  std::vector<double> intensitys;

  for (size_t ith = 0; ith < cloud->size(); ith++) {
    Eigen::Vector3d pt;
    VPoint p = cloud->at(ith);
    pt(0) = p.x;
    pt(1) = p.y;
    pt(2) = p.z;
    point3d.push_back(pt);
    intensitys.push_back(p.intensity);
  }
  std::cout << intensitys.size() << std::endl;

  Eigen::Vector2d RLRH;
  RLRH = calHist(intensitys);

  gray_zone(0) = ((rate - 1) * RLRH(0) + RLRH(1)) / rate;
  gray_zone(1) = (RLRH(0) + (rate - 1) * RLRH(1)) / rate;

  std::cout << "rate: " << rate << ", gray_zone: " << gray_zone.transpose()
            << std::endl;

  return gray_zone;
}

Eigen::Matrix4f
LidarCornersEst::transformbyPCA(VPointCloud::Ptr input_cloud,
                                VPointCloud::Ptr &output_cloud) {
  /// PCA 降维度
  Eigen::Vector4f pcaCentroid;
  pcl::compute3DCentroid(*input_cloud, pcaCentroid);
  Eigen::Matrix3f covariance;
  pcl::computeCovarianceMatrixNormalized(*input_cloud, pcaCentroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(
      covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
  eigenValuesPCA = eigen_solver.eigenvalues();

  std::cout << eigenVectorsPCA.col(1) << std::endl;

  eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

  std::cout << eigenVectorsPCA.col(2) << std::endl;

  Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
  transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
  transform.block<3, 1>(0, 3) =
      -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());

  pcl::PointCloud<VPoint>::Ptr PCA_cloud(new pcl::PointCloud<VPoint>);
  pcl::transformPointCloud(*input_cloud, *PCA_cloud, transform);

  //    cout << "PCA eigen values \n";
  std::cout << "eigenValuesPCA: " << eigenValuesPCA.transpose() << std::endl;
  std::cout << eigenVectorsPCA << std::endl;

  output_cloud = PCA_cloud;

  pca_matrix = transform;
  m_cloud_PCA = PCA_cloud;

  return transform;
}

void LidarCornersEst::PCA(VPointCloud::Ptr &input_cloud) {
  VPointCloud::Ptr PCA_cloud(new VPointCloud);
  transformbyPCA(input_cloud, PCA_cloud);

  m_gray_zone = get_gray_zone(input_cloud, 2.5);
}

bool LidarCornersEst::get_corners(std::vector<Eigen::Vector3d> &corners) {
  /// 拟合虚拟棋盘格
  bool useOutofBoard = true;
  bool topleftWhite = true;
  bool inverse_flag = true;

  VPointCloud corners_cloud;
  pcl::PointCloud<VPoint>::Ptr optim_cloud(new pcl::PointCloud<VPoint>);

  Optimization optim;
  Eigen::Vector3d theta_t(0, 0, 0);

  useOutofBoard = true;
  optim.get_theta_t(m_cloud_PCA, m_grid_size, m_gray_zone, topleftWhite,
                    m_grid_length, theta_t, useOutofBoard);
  useOutofBoard = false;
  optim.get_theta_t(m_cloud_PCA, m_grid_size, m_gray_zone, topleftWhite,
                    m_grid_length, theta_t, useOutofBoard);

  Eigen::Affine3f transf =
      pcl::getTransformation(0, theta_t(1), theta_t(2), theta_t(0), 0, 0);
  pcl::transformPointCloud(*m_cloud_PCA, *optim_cloud, transf);

  getPCDcorners(pca_matrix, transf.matrix(), corners_cloud, inverse_flag);

  m_cloud_optim = optim_cloud;
  m_cloud_corners = corners_cloud.makeShared();

  return true;
}

void LidarCornersEst::getPCDcorners(Eigen::Matrix4f transPCA,
                                    Eigen::Matrix4f transOptim,
                                    VPointCloud &corners, bool inverse) {
  Eigen::Vector2i grid_size = m_grid_size;
  // if (inverse) {
  //   grid_size(0) = m_grid_size(1);
  //   grid_size(1) = m_grid_size(0);
  // }
  corners.clear();
  std::vector<double> x_grid_arr;
  std::vector<double> y_grid_arr;

  double temp;
  for (int i = 1; i < grid_size(0); i++) {
    temp = (i - double(grid_size(0)) / 2.0) * m_grid_length;
    x_grid_arr.push_back(temp);
  }
  for (int i = 1; i < grid_size(1); i++) {
    temp = (i - double(grid_size(1)) / 2.0) * m_grid_length;
    y_grid_arr.push_back(temp);
  }

  // std::cout << x_grid_arr.size() << " " << y_grid_arr.size() << std::endl;

  for (unsigned int i = 0; i < x_grid_arr.size(); i++)
    for (unsigned int j = 0; j < y_grid_arr.size(); j++) {

      VPoint pt;
      pt.x = 0;
      pt.y = x_grid_arr.at(i);
      pt.z = y_grid_arr.at(j);
      pt.intensity = 50;
      corners.push_back(pt);
    }

  Eigen::Isometry3f T_optim = Eigen::Isometry3f::Identity();
  T_optim.rotate(transOptim.block<3, 3>(0, 0));
  T_optim.pretranslate(transOptim.block<3, 1>(0, 3));

  Eigen::Isometry3f T_PCA = Eigen::Isometry3f::Identity();
  T_PCA.rotate(transPCA.block<3, 3>(0, 0));
  T_PCA.pretranslate(transPCA.block<3, 1>(0, 3));

  Eigen::Isometry3f T_lidar2board = Eigen::Isometry3f::Identity();
  T_lidar2board = T_PCA.inverse() * T_optim.inverse();

  pcl::transformPointCloud(corners, corners, transOptim.inverse());
  pcl::transformPointCloud(corners, corners, transPCA.inverse());

  //    cout << "transOptim: " << endl;
  //    cout << transOptim << endl;
  //    cout << "transPCA: " << endl;
  //    cout << transPCA << endl;
}

void LidarCornersEst::CropBox(geometry_msgs::PointStamped selected_point,
                              VPointCloud::Ptr &in_cloud) {
  pcl::CropBox<VPoint> crop;
  crop.setInputCloud(in_cloud);
  crop.setMin(Eigen::Vector4f(selected_point.point.x - 1.0,
                              selected_point.point.y - 1.0,
                              selected_point.point.z - 0.5, 1.0));
  crop.setMax(Eigen::Vector4f(selected_point.point.x + 1.0,
                              selected_point.point.y + 1.0,
                              selected_point.point.z + 1.0, 1.0));
  crop.filter(*in_cloud);
}

bool LidarCornersEst::detectCornersFromLidar(VPointCloud::Ptr &in_cloud,
                                             VPointCloud::Ptr &corners_points) {
  if (in_cloud->points.size() < 5) {
    std::cout << "ROI点云过少，请重新选取！" << std::endl;
  }

  std::vector<Eigen::Vector3d> lidar_corner;
  // 进行角点检测
  PCA(in_cloud);
  get_corners(lidar_corner);
  corners_points = m_cloud_corners;

  return true;
}

void LidarCornersEst::show_pcd_corners(std::vector<Eigen::Vector3d> point3d,
                                       std::vector<cv::Point2d> point2d) {

  cv::Mat img_show(2200, 2000, CV_8UC3, cv::Scalar::all(125));

  for (uint i = 0; i < point3d.size(); i++) {
    cv::Point2d p;

    p.x = point3d.at(i)[0];
    p.y = point3d.at(i)[1];

    p.x = (p.x + 0.9) * 1000;
    p.y = (p.y + 0.2) * 1000;
    cv::circle(img_show, p, 2, cv::Scalar(255, 0, 0), 3);

    std::string str = std::to_string(i);
    cv::putText(img_show, str, p, 0.8, 1.0, cv::Scalar(0, 0, 255), 1);

    /// 绿色为像素点
    cv::circle(img_show, cv::Point2d(point2d.at(i).x * 2, point2d.at(i).y * 2),
               2, cv::Scalar(255, 0, 0), 3);
    cv::putText(img_show, str,
                cv::Point2d(point2d.at(i).x * 2, point2d.at(i).y * 2), 0.8, 1.0,
                cv::Scalar(0, 255, 0), 1);
  }

  cv::resize(img_show, img_show,
             cv::Size(img_show.cols * 0.5, img_show.rows * 0.5));
  cv::imshow("lidar and camera corners", img_show);
  cv::waitKey(0);
  cv::destroyAllWindows();
}




#include "Optimization.h"
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp> /// cv::solvePnPRansac

#include <CeresPnpError.h> /// CeresPnpError

Optimization::Optimization() {}

Eigen::Isometry3d Optimization::solvePose3d2dError(
    std::vector<Eigen::Vector3d> pts3d, std::vector<cv::Point2d> pts2d,
    cv::Mat K, Eigen::Vector3d &r_ceres, Eigen::Vector3d &t_ceres)

{
  Eigen::Vector4d camera;
  camera(0) = K.at<double>(0, 0);
  camera(1) = K.at<double>(0, 2);
  camera(2) = K.at<double>(1, 1);
  camera(3) = K.at<double>(1, 2);

  //    Eigen::Vector3d r_ceres(0,0,0);
  //    Eigen::Vector3d t_ceres(0,0,0);

  //    cv::Mat rvec, tvec;
  //    cv::solvePnPRansac( pts3d, pts2d, K, cv::Mat(), rvec, tvec, true,
  //    100, 4.0, 0.99); for(int i = 0; i < 3; i++)
  //    {
  //        r_ceres(i) = rvec.at<double>(i);
  //        t_ceres(i) = tvec.at<double>(i);
  //    }

  std::cout << "r_initial: " << r_ceres.transpose() << std::endl;
  std::cout << "t_initial: " << t_ceres.transpose() << std::endl;

  //优化
  ceres::Problem problem;
  for (unsigned int i = 0; i < pts3d.size(); i++) {
    ceres::CostFunction *cost_function = 0;

    cost_function = new ceres::AutoDiffCostFunction<Pose3d2dError, 2, 3, 3>(
        new Pose3d2dError(pts3d.at(i), pts2d.at(i), camera));

    problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1),
                             r_ceres.data(), t_ceres.data());
  }
  ceres::Solver::Options options;

  options.minimizer_type = ceres::TRUST_REGION;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.minimizer_progress_to_stdout = true;
  options.dogleg_type = ceres::SUBSPACE_DOGLEG;

  //    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  // options.max_num_iterations = 20;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";

  std::cout << "r_ceres: " << r_ceres.transpose() << std::endl;
  std::cout << "t_ceres: " << t_ceres.transpose() << std::endl;

  Eigen::Vector3d temp = r_ceres;
  double rad = temp.norm();
  temp.normalize();
  Eigen::AngleAxisd rotation_vector(rad, temp); //用旋转向量构造旋转向量！

  Eigen::Vector3d euler_angles = rotation_vector.matrix().eulerAngles(
      2, 1, 0); // ZYX顺序，即roll pitch yaw顺序
  std::cout << "yaw pitch roll = " << (euler_angles * 180 / 3.14).transpose()
            << std::endl;

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.rotate(rotation_vector.matrix());
  T.pretranslate(Eigen::Vector3d(t_ceres[0], t_ceres[1], t_ceres[2]));

  //    cout << "Eigen::Isometry3d \n" << T.matrix() << endl;
  return T;
}

void Optimization::get_theta_t(VPointCloud::Ptr cloud,
                               Eigen::Vector2i board_size,
                               Eigen::Vector2d gray_zone, bool topleftWhite,
                               double grid_length, Eigen::Vector3d &theta_t,
                               bool useOutofBoard) {

  //// theta_t 可以给初值
  Eigen::Vector3i black_gray_white(0, 0, 0); /// 统计
  ceres::Problem problem;
  VPoint temp;
  for (unsigned int i = 0; i < cloud->size(); i++) {
    temp = cloud->points[i];
    ceres::CostFunction *cost_function = 0;

    bool laser_white;
    if (temp.intensity < gray_zone[0]) {
      laser_white = false;
      black_gray_white(0)++;
    } else if (temp.intensity > gray_zone[1]) {
      laser_white = true;
      black_gray_white(2)++;
    } else {
      black_gray_white(1)++;
      continue;
    }

    Eigen::Vector2d laserPoint(temp.y, temp.z); // = point2d.at(i);
    Eigen::Matrix2d sqrtPrecisionMat = Eigen::Matrix2d::Identity();

    cost_function = new ceres::AutoDiffCostFunction<VirtualboardError, 1, 3>(
        new VirtualboardError(board_size, topleftWhite, grid_length,
                              laser_white, useOutofBoard, laserPoint,
                              sqrtPrecisionMat));

    // new ceres::CauchyLoss(0.5)
    problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1),
                             theta_t.data());
  }
  //    std::cout << "black_gray_white: " << black_gray_white.transpose() <<
  //    std::endl;
  ceres::Solver::Options options;

  options.minimizer_type = ceres::TRUST_REGION;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.minimizer_progress_to_stdout = false;
  options.dogleg_type = ceres::SUBSPACE_DOGLEG;

  //    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  //    options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";

  std::cout << theta_t.transpose() << std::endl;
}

void Optimization::get_board_corner(cv::Size boardSize, double squareSize,
                                    std::vector<cv::Point3d> &bBoardCorner) {
  double board_z_axis = 1.1;
  const cv::Point3d pointO(boardSize.width * squareSize / 2.0,
                           boardSize.height * squareSize / 2.0, board_z_axis);

  for (int i = 0; i < boardSize.height; i++) {
    for (int j = 0; j < boardSize.width; j++) {
      cv::Point3d p;
      p.x = (j - pointO.x) * squareSize;
      p.y = (i - pointO.y) * squareSize;
      p.z = board_z_axis;
      bBoardCorner.push_back(p);
    }
  }
}

Eigen::Isometry3d Optimization::solvePnP(std::vector<cv::Point2f> corners,
                                         cv::Mat camK, cv::Size boardSize,
                                         double squareSize) {
  std::shared_ptr<Camera> camera(new Camera);

  double fx = camK.at<double>(0, 0);
  double cx = camK.at<double>(0, 2);
  double fy = camK.at<double>(1, 1);
  double cy = camK.at<double>(1, 2);
  camera->set(fx, fy, cx, cy);

  std::vector<cv::Point3d> bBoardCorner;
  get_board_corner(boardSize, squareSize, bBoardCorner);

  ceres::Problem problem;

  Eigen::Matrix<double, 6, 1> se3;
  se3 << 0, 0, 0, 0, 0, 0;
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  for (size_t i = 0; i < bBoardCorner.size(); ++i) {

    Eigen::Vector3d pt(bBoardCorner.at(i).x, bBoardCorner.at(i).y,
                       bBoardCorner.at(i).z);
    Eigen::Vector2d uv(corners.at(i).x, corners.at(i).y);

    ceres::CostFunction *costFun =
        new CeresPnpError(pt, uv, information, camera);
    problem.AddResidualBlock(costFun, new ceres::HuberLoss(0.5), se3.data());
  }

  problem.SetParameterization(se3.data(), new SE3Parameterization());

  ceres::Solver::Options options;
  options.minimizer_type = ceres::TRUST_REGION;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::DOGLEG;
  // options.minimizer_progress_to_stdout = true;
  options.dogleg_type = ceres::SUBSPACE_DOGLEG;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";

  Sophus::SE3 T = Sophus::SE3::exp(se3);

  //  std::cout << T.log() << std::endl;    /// 前三维不是位移！！后三维旋转向量
  //  cout << T.so3() << endl;    /// 旋转向量
  //  Eigen::Vector3d euler_angles = (T.rotation_matrix()).eulerAngles ( 2,1,0
  //  ); std::cout << "CeresPnp:" << (euler_angles*180.0/3.14).transpose() <<" "
  //            << T.translation().transpose() << std::endl;

  Eigen::Isometry3d T_board2cam = Eigen::Isometry3d::Identity();
  T_board2cam.rotate(T.rotation_matrix());
  T_board2cam.pretranslate(T.translation()); /// 位移

  //  std::cout << T_board2cam.matrix() << std::endl;

  return T_board2cam;
}

#include "ImageCornersEst.h"
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp> /// cv::findChessboardCorners
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> /// cv::undistort

ImageCornersEst::ImageCornersEst(std::string cam_yaml) {

  cv::FileStorage fs;
  fs.open(cam_yaml, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    std::cerr << "can not open " << cam_yaml << std::endl;
    return;
  }
  fs["CameraMat"] >> camK;
  fs["DistCoeff"] >> distort_param;
  fs["ImageSize"]["width"] >> m_image_size.width;
  fs["ImageSize"]["height"] >> m_image_size.height;
  m_grid_length = static_cast<double>(fs["grid_length"]);
  fs["ChessBoardSize"] >> m_board_size;
  std::cout << "image size:" << m_image_size.width << "," << m_image_size.height
            << std::endl;

  fs.release();

  m_fx = camK.at<double>(0, 0);
  m_cx = camK.at<double>(0, 2);
  m_fy = camK.at<double>(1, 1);
  m_cy = camK.at<double>(1, 2);
  std::cout << "m_fx" << m_fx << "m_cx" << m_cx << "m_fy" << m_fy << "m_cy"
            << m_cy << std::endl;

  std::cout << "camK: \n" << camK << std::endl;
  std::cout << "dist: \n" << distort_param << std::endl;
  std::cout << "grid_length: " << m_grid_length << std::endl;
  std::cout << "corner_in_x: " << m_board_size.width << std::endl;
  std::cout << "corner_in_y: " << m_board_size.height << std::endl;
}

bool ImageCornersEst::getRectifyParam(std::string cam_yaml) { return true; }

void ImageCornersEst::undistort_image(cv::Mat image, cv::Mat &rectify_image) {
  cv::undistort(image, rectify_image, camK, distort_param, camK);
}

bool ImageCornersEst::spaceToPlane(Eigen::Vector3d P_w, Eigen::Vector2d &P_cam,
                                   double dis) {
  Eigen::Vector3d P_c = m_R * P_w + m_t;
  //        Eigen::Vector3d P_c = P_w;
  if (P_c[2] < 0 || P_c[2] > dis) {
    return false;
  }
  // Transform to model plane
  double u = P_c[0] / P_c[2];
  double v = P_c[1] / P_c[2];

  P_cam(0) = m_fx * u + m_cx;
  P_cam(1) = m_fy * v + m_cy;

  if (P_cam(0) > 0 && P_cam(0) < m_image_size.width && P_cam(1) > 0 &&
      P_cam(1) < m_image_size.height) {
    return true;
  } else
    return false;
}

void ImageCornersEst::show_calib_result(std::vector<Eigen::Vector3d> point3d,
                                        std::vector<cv::Point2d> point2d,
                                        cv::Mat &image) {
  float errorSum = 0.0f;
  float errorMax = 0;
  for (unsigned int i = 0; i < point3d.size(); i++) {
    Eigen::Vector3d Pw = point3d.at(i);
    Eigen::Vector2d Pcam;
    if (spaceToPlane(Pw, Pcam)) {

      cv::Point2d pEst(Pcam[0], Pcam[1]);
      cv::Point2d pObs = point2d.at(i);

      cv::circle(image, pEst, 1, cv::Scalar(0, 0, 255), 2); // r
      cv::circle(image, pObs, 1, cv::Scalar(255, 0, 0), 2); // b

      float error = cv::norm(pObs - pEst);

      errorSum += error;
      if (error > errorMax) {
        errorMax = error;
      }
    }
  }

  std::ostringstream oss;
  oss << "Reprojection error: avg = " << errorSum / 35.0
      << "   max = " << errorMax;

  cv::putText(image, oss.str(), cv::Point(10, image.rows - 20),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 0.5,
              cv::LINE_AA);
  std::cout << "  " << oss.str() << std::endl;
}

void ImageCornersEst::split(std::string &s, std::string &delim,
                            std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (index - last > 0) {
    ret.push_back(s.substr(last, index - last));
  }
}

void ImageCornersEst::read_cam_corners(std::string filename, int num,
                                       std::vector<cv::Point2d> &point2d) {
  std::vector<std::vector<cv::Point2d>> Corners;

  std::ifstream infile;
  infile.open(filename);

  std::string s;
  int counter = 0;
  std::string dlim = " ";
  while (std::getline(infile, s)) {
    std::vector<std::string> slist;
    split(s, dlim, slist);
    std::vector<cv::Point2d> corner;
    for (uint i = 0; i < slist.size(); i++) {
      std::stringstream ss;
      ss << slist[i];
      cv::Point2d p;
      ss >> p.x;
      p.y = 0;
      corner.push_back(p);
      counter++;
    }
    Corners.push_back(corner);

    if (counter >= num)
      break;
  }

  counter = 0;
  while (std::getline(infile, s)) {
    std::vector<std::string> slist;
    split(s, dlim, slist);
    for (uint i = 0; i < slist.size(); i++) {
      std::stringstream ss;
      ss << slist[i];
      ss >> Corners[counter][i].y;
    }
    counter++;
  }

  infile.close();

  /// ok
  if (Corners.size() != m_board_size.height) {
    for (unsigned int w = 0; w < Corners.at(0).size(); w++)
      for (unsigned int h = 0; h < Corners.size(); h++)
        point2d.push_back(Corners[h][w]);

  } else {
    for (unsigned int h = 0; h < Corners.size(); h++)
      for (unsigned int w = 0; w < Corners.at(0).size(); w++)
        point2d.push_back(Corners[h][w]);
  }

  // std::cout << " camera corner size: " << point2d.size() << std::endl;
}

void ImageCornersEst::read_lidar_corners(
    std::string filename, int num, std::vector<Eigen::Vector3d> &point3d) {

  std::ifstream infile;

  int counter = 0;
  infile.open(filename.c_str());
  while (!infile.eof() && counter < num) {
    float x, y, z;
    infile >> x >> y >> z;
    point3d.push_back(Eigen::Vector3d(x, y, z));

    counter++;
  }
  infile.close();

  // std::cout << " lidar  corner size: " << point3d.size() << std::endl;
}

bool ImageCornersEst::detectCornersFromCam(
    cv::Mat input_image, std::vector<cv::Point2f> &image_corners,
    cv::Mat &output_image) {
  cv::Mat src_image;
  cv::cvtColor(input_image, src_image, cv::COLOR_BGR2GRAY);
  bool whether_found =
      cv::findChessboardCorners(src_image, m_board_size, image_corners);
  if (whether_found) {
    cv::cornerSubPix(
        src_image, image_corners, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                         0.01));
    // 绘制检测到的角点
    cv::drawChessboardCorners(src_image, m_board_size, cv::Mat(image_corners),
                              whether_found);

    output_image = src_image.clone();
    std::cout << "角点检测成功!" << std::endl;
    return true;
  }
  std::cout << "角点检测失败!" << std::endl;
  return false;
}

void ImageCornersEst::extrinsic2txt(std::string savefile,
                                    Eigen::Matrix4d lidar2cam) {
  // 打开文件，以文本模式写入
  std::ofstream outfile(savefile.c_str());

  // 遍历矩阵的每个元素，并按行写入文件
  if (outfile.is_open()) {
    for (int i = 0; i < lidar2cam.rows(); ++i) {
      for (int j = 0; j < lidar2cam.cols(); ++j) {
        outfile << lidar2cam(i, j); // 输出矩阵元素
        if (j < lidar2cam.cols() - 1) {
          outfile << " "; // 在行的元素之间加空格
        }
      }
      outfile << std::endl; // 每行结束后换行
    }
  }

  outfile.close(); // 关闭文件
}

void ImageCornersEst::txt2extrinsic(std::string filepath) {
  Eigen::Matrix4d lidar2cam;
  std::ifstream infile(filepath, std::ios_base::binary);
  infile.read((char *)&lidar2cam, sizeof(Eigen::Matrix4d));
  infile.close();
  std::cout << "lidar2cam\n" << lidar2cam << std::endl;

  Eigen::Isometry3d T_lidar2cam_temp = Eigen::Isometry3d::Identity();
  T_lidar2cam_temp.rotate(lidar2cam.block<3, 3>(0, 0));
  T_lidar2cam_temp.pretranslate(lidar2cam.block<3, 1>(0, 3));

  m_R = lidar2cam.block<3, 3>(0, 0);
  m_t = lidar2cam.block<3, 1>(0, 3);

  T_lidar2cam = T_lidar2cam_temp;

  std::cout << "cam2lidar: \n"
            << T_lidar2cam_temp.inverse().matrix() << std::endl;
}

void ImageCornersEst::HSVtoRGB(int h, int s, int v, unsigned char *r,
                               unsigned char *g, unsigned char *b) {
  // convert from HSV/HSB to RGB color
  // R,G,B from 0-255, H from 0-260, S,V from 0-100
  // ref http://colorizer.org/

  // The hue (H) of a color refers to which pure color it resembles
  // The saturation (S) of a color describes how white the color is
  // The value (V) of a color, also called its lightness, describes how dark the
  // color is

  int i;

  float RGB_min, RGB_max;
  RGB_max = v * 2.55f;
  RGB_min = RGB_max * (100 - s) / 100.0f;

  i = h / 60;
  int difs = h % 60; // factorial part of h

  // RGB adjustment amount by hue
  float RGB_Adj = (RGB_max - RGB_min) * difs / 60.0f;

  switch (i) {
  case 0:
    *r = RGB_max;
    *g = RGB_min + RGB_Adj;
    *b = RGB_min;
    break;
  case 1:
    *r = RGB_max - RGB_Adj;
    *g = RGB_max;
    *b = RGB_min;
    break;
  case 2:
    *r = RGB_min;
    *g = RGB_max;
    *b = RGB_min + RGB_Adj;
    break;
  case 3:
    *r = RGB_min;
    *g = RGB_max - RGB_Adj;
    *b = RGB_max;
    break;
  case 4:
    *r = RGB_min + RGB_Adj;
    *g = RGB_min;
    *b = RGB_max;
    break;
  default: // case 5:
    *r = RGB_max;
    *g = RGB_min;
    *b = RGB_max - RGB_Adj;
    break;
  }
}

void ImageCornersEst::check_order_cam(std::vector<cv::Point2d> &point2d,
                                      cv::Size boardSize) {
  /// check point2D
  if (point2d.at(0).y > point2d.at(boardSize.width + 1).y) {
    for (int h = 0; h < boardSize.height / 2; h++) {
      int front = boardSize.width * h;
      int end = boardSize.width * (boardSize.height - 1 - h);
      for (int w = 0; w < boardSize.width; w++) {
        cv::Point2d p = point2d.at(front + w);
        point2d.at(front + w) = point2d.at(end + w);
        point2d.at(end + w) = p;
      }
    }
  }
  if (point2d.at(0).x > point2d.at(1).x) {
    for (int h = 0; h < boardSize.height; h++) {
      int front = boardSize.width * h;
      for (int w = 0; w < boardSize.width / 2; w++) {
        cv::Point2d p = point2d.at(front + w);
        point2d.at(front + w) = point2d.at(front + boardSize.width - 1 - w);
        point2d.at(front + boardSize.width - 1 - w) = p;
      }
    }
  }
}

void ImageCornersEst::check_order_lidar(std::vector<Eigen::Vector3d> &point3d,
                                        cv::Size boardSize) {
  /// check point3D
  if (point3d.at(0)[1] > point3d.at(boardSize.width + 1)[1]) {
    for (int h = 0; h < boardSize.height / 2; h++) {
      int front = boardSize.width * h;
      int end = boardSize.width * (boardSize.height - 1 - h);
      for (int w = 0; w < boardSize.width; w++) {
        Eigen::Vector3d p = point3d.at(front + w);
        point3d.at(front + w) = point3d.at(end + w);
        point3d.at(end + w) = p;
      }
    }
  }
  if (point3d.at(0)[0] > point3d.at(1)[0]) {
    for (int h = 0; h < boardSize.height; h++) {
      int front = boardSize.width * h;
      for (int w = 0; w < boardSize.width / 2; w++) {
        Eigen::Vector3d p = point3d.at(front + w);
        point3d.at(front + w) = point3d.at(front + boardSize.width - 1 - w);
        point3d.at(front + boardSize.width - 1 - w) = p;
      }
    }
  }
}

