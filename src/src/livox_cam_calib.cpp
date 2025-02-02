/*
 * @Author: huangwei barry.huangw@gmail.com
 * @Date: 2024-09-13 15:22:51
 * @LastEditors: huangwei barry.huangw@gmail.com
 * @LastEditTime: 2024-10-24 14:30:49
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
        std::string projected_image_path = res_save_file_name_ +
                                           "projectedImage_" +
                                           std::to_string(i) + ".png";
        cv::imwrite(projected_image_path, image);
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
