#include <pcl/point_types.h>
 
struct EIGEN_ALIGN16 PointXYZRGBI
{
    PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
    PCL_ADD_RGB;
    float i;     //intensity
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBI,
                                  (float,x,x)
                                  (float,y,y)
                                  (float,z,z)
                                  (uint8_t,r,r)
                                  (uint8_t,g,g)
                                  (uint8_t,b,b)
                                  (float,i,i)
)



#ifndef CERESPNPERROR_H
#define CERESPNPERROR_H

#include "sophus/se3.h"
#include "sophus/so3.h"
#include <ceres/ceres.h>

class Camera {
public:
  Camera() {}
  Camera(double fx, double fy, double cx, double cy) {
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
  }

  void set(double fx, double fy, double cx, double cy) {
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
  }

  Eigen::Vector2d project(double x, double y, double z) {
    Eigen::Vector2d uv;
    uv(0) = fx_ * x / z + cx_;
    uv(1) = fy_ * y / z + cy_;

    return uv;
  }
  Eigen::Vector2d project(Eigen::Vector3d point) {
    return project(point(0), point(1), point(2));
  }

  double fx_;
  double fy_;
  double cx_;
  double cy_;
};

/**
 * @brief The CeresPnpError class
 * 继承ceres::SizedCostFunction，
 * 声明误差维度、优化量维度1、(优化量维度2、...)
 * 这里只声明了优化量维度1，之后的只有if(jacobians[0])会是true
 */
class CeresPnpError : public ceres::SizedCostFunction<2, 6> {
public:
  CeresPnpError(Eigen::Vector3d &pt, Eigen::Vector2d &uv,
                Eigen::Matrix<double, 2, 2> &information,
                std::shared_ptr<Camera> cam)
      : pt_(pt), uv_(uv), cam_(cam) {
    // printf("index = %d\n", index++);
    Eigen::LLT<Eigen::Matrix<double, 2, 2>> llt(information);
    sqrt_information_ = llt.matrixL();
  }
  virtual ~CeresPnpError() {}
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(parameters[0]);
    Sophus::SE3 T = Sophus::SE3::exp(lie);

    // std::cout << T.matrix3x4() << std::endl;

    Eigen::Vector3d P = T * pt_;
    Eigen::Vector2d uv = cam_->project(P);
    Eigen::Vector2d err = uv - uv_;
    err = sqrt_information_ * err;

    residuals[0] = err(0);
    residuals[1] = err(1);

    Eigen::Matrix<double, 2, 6> Jac = Eigen::Matrix<double, 2, 6>::Zero();
    Jac(0, 0) = cam_->fx_ / P(2);
    Jac(0, 2) = -P(0) / P(2) / P(2) * cam_->fx_;
    Jac(0, 3) = Jac(0, 2) * P(1);
    Jac(0, 4) = cam_->fx_ - Jac(0, 2) * P(0);
    Jac(0, 5) = -Jac(0, 0) * P(1);

    Jac(1, 1) = cam_->fy_ / P(2);
    Jac(1, 2) = -P(1) / P(2) / P(2) * cam_->fy_;
    Jac(1, 3) = -cam_->fy_ + Jac(1, 2) * P(1);
    Jac(1, 4) = -Jac(1, 2) * P(0);
    Jac(1, 5) = Jac(1, 1) * P(0);

    Jac = sqrt_information_ * Jac;

    int k = 0;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (k >= 12)
          return false;
        if (jacobians) {
          if (jacobians[0])
            jacobians[0][k] = Jac(i, j);
        }
        k++;
      }
    }

    // printf("jacobian ok!\n");

    return true;
  }

public:
  Eigen::Vector3d pt_;
  Eigen::Vector2d uv_;
  std::shared_ptr<Camera> cam_;
  Eigen::Matrix<double, 2, 2> sqrt_information_;
  static int index;
};

int CeresPnpError::index = 0;

class CERES_EXPORT SE3Parameterization : public ceres::LocalParameterization {
public:
  SE3Parameterization() {}
  virtual ~SE3Parameterization() {}
  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 6; }
  virtual int LocalSize() const { return 6; }
};

bool SE3Parameterization::ComputeJacobian(const double *x,
                                          double *jacobian) const {
  ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
  return true;
}

bool SE3Parameterization::Plus(const double *x, const double *delta,
                               double *x_plus_delta) const {
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

  Sophus::SE3 T = Sophus::SE3::exp(lie);
  Sophus::SE3 delta_T = Sophus::SE3::exp(delta_lie);
  Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

  for (int i = 0; i < 6; ++i)
    x_plus_delta[i] = x_plus_delta_lie(i, 0);

  return true;
}

#endif // CERESPNPERROR_H


/*
 * @Author: huangwei barry.huangw@gmail.com
 * @Date: 2024-10-21 17:01:52
 * @LastEditors: huangwei barry.huangw@gmail.com
 * @LastEditTime: 2024-10-22 10:43:10
 * @FilePath: /livox_cam_calib/src/include/common.h
 * @Description: 工具类函数
 */

#ifndef COMMON_HPP_
#define COMMON_HPP_
#include "pcl_ros/transforms.h"
#include "types.h"
#include <boost/thread.hpp>
#include <cstdio>
#include <ctime>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <livox_cam_calib/boundsConfig.h>
#include <opencv2/core.hpp>
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
#include <sys/stat.h>
#include <sys/types.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <vision_msgs/Detection2DArray.h>

namespace common {
class Common {
private:
  cv::Size corner_size_;

public:
  typedef std::shared_ptr<Common> Ptr;
  Common(cv::Size chessboard_size) { corner_size_ = chessboard_size; };
  virtual ~Common(){};

  /// 这个是为了让相机和激光坐标系方向一致
  Eigen::Isometry3d get_lidar2cam_axis_roughly() {

    Eigen::Matrix3d R_lidarToCamera;
    R_lidarToCamera << 0, -1, 0, 0, 0, -1, 1, 0, 0;

    Eigen::Isometry3d T_lidar2cam = Eigen::Isometry3d::Identity();
    T_lidar2cam.rotate(R_lidarToCamera);

    return T_lidar2cam;
  }

  void drawCornersOnImage(cv::Mat &image,
                          std::vector<cv::Point2f> &corners_vector) {
    for (size_t i = 0; i < corners_vector.size(); ++i) {
      const cv::Point2f &point = corners_vector[i];

      // 在图像上绘制角点
      cv::circle(image, point, 5, cv::Scalar(0, 255, 0), -1); // 绘制绿色圆点

      // 构造坐标文本
      std::ostringstream oss;
      oss << "(" << static_cast<int>(point.x) << ", "
          << static_cast<int>(point.y) << ")";
      std::string text = oss.str();

      // 根据角点的索引变化调整文本位置，避免重叠
      cv::Point2f text_offset;
      if (i % 4 == 0) {
        text_offset = cv::Point2f(10, -10); // 右上方
      } else if (i % 4 == 1) {
        text_offset = cv::Point2f(-10, 10); // 左下方
      } else if (i % 4 == 2) {
        text_offset = cv::Point2f(10, 10); // 右下方
      } else {
        text_offset = cv::Point2f(-10, -10); // 左上方
      }

      // 在角点附近打印坐标
      cv::putText(image, text, point + text_offset, cv::FONT_HERSHEY_SIMPLEX,
                  0.45, cv::Scalar(255, 0, 0),
                  1); // 红色文本
    }
  }

  void drawCoordinateAxes(cv::Mat &image, const cv::Point &origin,
                          double length) {
    // 绘制X轴（红色）
    cv::line(image, origin, cv::Point(origin.x + length, origin.y),
             cv::Scalar(0, 0, 255), 2); // 红色
    // 绘制Y轴（绿色）
    cv::line(image, origin, cv::Point(origin.x, origin.y - length),
             cv::Scalar(0, 255, 0), 2); // 绿色
    // 绘制Z轴（蓝色）
    cv::line(image, origin,
             cv::Point(origin.x - length / 2, origin.y - length / 2),
             cv::Scalar(255, 0, 0), 2); // 蓝色

    // 绘制坐标轴箭头
    cv::arrowedLine(image, origin, cv::Point(origin.x + length, origin.y),
                    cv::Scalar(0, 0, 255), 2);
    cv::arrowedLine(image, origin, cv::Point(origin.x, origin.y - length),
                    cv::Scalar(0, 255, 0), 2);
    cv::arrowedLine(image, origin,
                    cv::Point(origin.x - length / 2, origin.y - length / 2),
                    cv::Scalar(255, 0, 0), 2);

    // 添加坐标轴标签
    cv::putText(image, "X", cv::Point(origin.x + length, origin.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::putText(image, "Y", cv::Point(origin.x, origin.y - length),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::putText(image, "Z",
                cv::Point(origin.x - length / 2, origin.y - length / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
  }

  void savePoint3dToFile(const std::string &filename,
                         const std::vector<Eigen::Vector3d> &points) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
    }

    for (const auto &point : points) {
      outfile << point.x() << " " << point.y() << " " << point.z() << "\n";
    }

    outfile.close();
  }

  void savePoint2dToFile(const std::string &filename,
                         const std::vector<cv::Point2d> &points) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
    }

    for (const auto &point : points) {
      outfile << point.x << " " << point.y << "\n";
    }

    outfile.close();
  }

  void save_corners2txt(VPointCloud::Ptr cloud, std::string filename) {
    std::ofstream outfile(filename.c_str(), std::ios_base::trunc);

    VPoint temp;
    for (unsigned int i = 0; i < cloud->size(); i++) {
      temp = cloud->points[i];
      outfile << temp.x << " " << temp.y << " " << temp.z << endl;
    }
    outfile.close();
  }

  void save_corners2txt(std::vector<cv::Point2f> corners,
                        std::string filename) {
    // 保存数据到文件
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "无法打开文件!" << std::endl;
      return;
    }
    // 写入数据，按行保存
    for (int r = 0; r < corner_size_.height; ++r) {
      for (int c = 0; c < corner_size_.width; ++c) {
        int index = r * corner_size_.width + c;
        if (index < corners.size()) {
          file << corners[index].x << " " << corners[index].y << " ";
        }
      }
      file << "\n"; // 换行
    }

    file.close();

    return;
  }

  void publish_cloud(ros::Publisher &pub, VPointCloud::Ptr cloud_to_pub) {

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_to_pub, cloud_msg);
    cloud_msg.header.stamp = ros::Time::now();
    pub.publish(cloud_msg);
  }

  void publish_pointcloud(const ros::Publisher &cloud_pub,
                          const std_msgs::Header &header,
                          const VPointCloud::ConstPtr &cloud) {
    if (cloud_pub.getNumSubscribers()) {
      sensor_msgs::PointCloud2 out_pc;
      pcl::toROSMsg(*cloud, out_pc);
      out_pc.header = header;
      cloud_pub.publish(out_pc);
    }
  }
  void publishImage(const ros::Publisher &image_pub,
                    const std_msgs::Header &header, const cv::Mat image) {
    cv_bridge::CvImage output_image;
    output_image.header.frame_id = header.frame_id;
    output_image.encoding = sensor_msgs::image_encodings::MONO8;
    output_image.image = image;
    image_pub.publish(output_image);
  }

  void publishCloudtoShow(const ros::Publisher &cloudtoshow_pub,
                          const std_msgs::Header &header,
                          const VPointCloud::Ptr &cloud) {
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud, output_msg);
    output_msg.header = header;
    cloudtoshow_pub.publish(output_msg);
  }
};

} // namespace common
#endif // COMMON_HPP_


/*
 * @Author: huangwei barry.huangw@gmail.com
 * @Date: 2024-10-16 17:12:59
 * @LastEditors: huangwei barry.huangw@gmail.com
 * @LastEditTime: 2024-10-22 10:26:08
 * @FilePath: /livox_cam_calib/src/include/ImageCornersEst.h
 * @Description: 相机的工具类函数角点检测，投影等
 */
#ifndef IMAGECORNERSEST_H_
#define IMAGECORNERSEST_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/core.hpp>

#include <memory> /// std::shared_ptr

#include <fstream>

class ImageCornersEst {

public:
  typedef std::shared_ptr<ImageCornersEst> Ptr;

  ImageCornersEst(std::string cam_yaml);

  bool getRectifyParam(std::string cam_yaml);

  void undistort_image(cv::Mat image, cv::Mat &rectify_image);
  void undistort_stereo_image(cv::Mat image, cv::Mat &rectify_image);

  bool findCorners(cv::Mat image, bool undist = true);

  bool detectCornersFromCam(cv::Mat &output_image);

  void setRt(Eigen::Matrix3d R, Eigen::Vector3d t) {
    m_R = R;
    m_t = t;

    T_lidar2cam.translate(t);
    T_lidar2cam.rotate(R);
  }
  bool spaceToPlane(Eigen::Vector3d P_w, Eigen::Vector2d &P_cam,
                    double dis = 6);

  void show_calib_result(std::vector<Eigen::Vector3d> point3d,
                         std::vector<cv::Point2d> point2d, cv::Mat &image);

  void split(std::string &s, std::string &delim, std::vector<std::string> &ret);
  void read_cam_corners(std::string filename, int num,
                        std::vector<cv::Point2d> &point2d);
  void read_lidar_corners(std::string filename, int num,
                          std::vector<Eigen::Vector3d> &point3d);

  bool detectCornersFromCam(cv::Mat input_image,
                            std::vector<cv::Point2f> &image_corners,
                            cv::Mat &output_image);

  void extrinsic2txt(std::string savefile, Eigen::Matrix4d lidar2cam);
  void txt2extrinsic(std::string filepath);
  void HSVtoRGB(int h, int s, int v, unsigned char *r, unsigned char *g,
                unsigned char *b);

  void check_order_cam(std::vector<cv::Point2d> &point2d, cv::Size boardSize);
  void check_order_lidar(std::vector<Eigen::Vector3d> &point3d,
                         cv::Size boardSize);

  std::vector<cv::Point2f> corners_now;
  cv::Mat image_now;
  cv::Mat image_chessboard;
  cv::Size m_board_size; // 指定标定板角点数
  double m_grid_length;

  bool m_verbose;
  double m_fx, m_fy, m_cx, m_cy;
  cv::Mat camK, distort_param;

  cv::Mat Kr, dr;
  cv::Mat car_R, car_t;

  Eigen::Matrix3d m_R;
  Eigen::Vector3d m_t;
  Eigen::Isometry3d T_lidar2cam;

private:
  cv::Size m_image_size;
};

#endif // IMAGECORNERSEST_H_


/*
 * @Author: huangwei barry.huangw@gmail.com
 * @Date: 2024-10-15 16:30:50
 * @LastEditors: huangwei barry.huangw@gmail.com
 * @LastEditTime: 2024-10-22 10:47:28
 * @FilePath: /livox_cam_calib/src/include/LidarCornersEst.h
 * @Description: 点云的棋盘格角点检测
 */
#ifndef LIDARCORNERSEST_H_
#define LIDARCORNERSEST_H_

#include "Optimization.h"
#include "common.hpp"
#include "types.h"
#include <iostream>
#include <memory> /// std::shared_ptr
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
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

class LidarCornersEst {

public:
  typedef std::shared_ptr<LidarCornersEst> Ptr;

  LidarCornersEst() {
    m_cloud_ROI = VPointCloud::Ptr(new VPointCloud);
    m_cloud_chessboard = VPointCloud::Ptr(new VPointCloud);
    m_cloud_PCA = VPointCloud::Ptr(new VPointCloud);
    m_cloud_optim = VPointCloud::Ptr(new VPointCloud);
    m_cloud_corners = VPointCloud::Ptr(new VPointCloud);
  }

  void set_chessboard_param(cv::Size grid_info, double grid_legnth);

  Eigen::Vector2d calHist(std::vector<double> datas);
  Eigen::Vector2d get_gray_zone(VPointCloud::Ptr &cloud, double rate);
  Eigen::Matrix4f transformbyPCA(VPointCloud::Ptr input_cloud,
                                 VPointCloud::Ptr &output_cloud);

  void PCA(VPointCloud::Ptr &input_cloud);

  bool get_corners(std::vector<Eigen::Vector3d> &corners);

  void color_by_gray_zone(VPointCloud::Ptr cloud, Eigen::Vector2d gray_zone,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr &rgbcloud);

  void getPCDcorners(Eigen::Matrix4f transPCA, Eigen::Matrix4f transOptim,
                     VPointCloud &corners, bool inverse = false);

  void cornerCloud2vector(VPointCloud corner_cloud,
                          std::vector<Eigen::Vector3d> &corners);

  void CropBox(geometry_msgs::PointStamped selected_point,
               VPointCloud::Ptr &in_cloud);

  bool detectCornersFromLidar(VPointCloud::Ptr &in_cloud,
                              VPointCloud::Ptr &corners_points);

  void show_pcd_corners(std::vector<Eigen::Vector3d> point3d,
                        std::vector<cv::Point2d> point2d);
  VPoint m_click_point;
  VPointCloud::Ptr m_cloud_ROI;
  VPointCloud::Ptr m_cloud_chessboard;
  VPointCloud::Ptr m_cloud_PCA;
  VPointCloud::Ptr m_cloud_optim;
  VPointCloud::Ptr m_cloud_corners;

private:
  Eigen::Matrix4f pca_matrix;
  Eigen::Vector3f eigenValuesPCA;

  Eigen::Vector2d m_gray_zone;

  double m_grid_length;
  Eigen::Vector2i m_grid_size; // 指定标定板角点数
};

#endif


#ifndef OPTIMIZATION_H_
#define OPTIMIZATION_H_

#include "sophus/se3.h"
#include "sophus/so3.h"
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h> /// ceres::AngleAxisRotatePoint
#include <opencv2/core.hpp>

#include "common.hpp"
#include "types.h"

class VirtualboardError {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VirtualboardError(const Eigen::Vector2i &board_size, const bool topleftWhite,
                    const double grid_length, const bool laser_white,
                    const bool useOutofBoard, const Eigen::Vector2d &laserPoint,
                    const Eigen::Matrix2d &sqrtPrecisionMat)
      : m_board_size(board_size), m_topleftWhite(topleftWhite),
        m_grid_length(grid_length), m_laser_white(laser_white),
        m_useOutofBoard(useOutofBoard), m_laserPoint(laserPoint),
        m_sqrtPrecisionMat(sqrtPrecisionMat) {}
  template <typename T>
  bool operator()(const T *const theta_t, T *residuals) const {
    Eigen::Matrix<T, 2, 1> P = m_laserPoint.cast<T>();

    const T angle_axis[3] = {theta_t[0], T(0), T(0)};
    const T pt3[3] = {T(0), P(0), P(1)};
    T result[3];
    ceres::AngleAxisRotatePoint(angle_axis, pt3, result);
    result[1] = result[1] + theta_t[1];
    result[2] = result[2] + theta_t[2];

    T i = (result[1] + T(m_board_size(0)) * T(m_grid_length) / T(2.0)) /
          T(m_grid_length);
    T j = (result[2] + T(m_board_size(1)) * T(m_grid_length) / T(2.0)) /
          T(m_grid_length);
    // in board
    if (i > T(0) && i < T(m_board_size(0)) && j > T(0) &&
        j < T(m_board_size(1))) {
      T ifloor = floor(i);
      T jfloor = floor(j);

      T ii = floor(ifloor / T(2)) * T(2);
      T jj = floor(jfloor / T(2)) * T(2);

      bool White = !m_topleftWhite;
      if (ifloor == ii && jfloor == jj) // 都是偶数
        White = m_topleftWhite;
      if (ifloor != ii && jfloor != jj) // 都是奇数
        White = m_topleftWhite;

      /// 颜色一致，无误差
      if (m_laser_white == White) {
        residuals[0] = T(0);
      } else {
        T iceil = ceil(i);
        T jceil = ceil(j);
        T ierror, jerror;
        if (i - ifloor > T(0.5))
          ierror = iceil - i;
        else
          ierror = i - ifloor;

        if (j - jfloor > T(0.5))
          jerror = jceil - j;
        else
          jerror = j - jfloor;

        residuals[0] = ierror + jerror;
      }

    }
    // out of board
    else if (m_useOutofBoard) {

      T ierror; /*= min( abs(i-T(0)) , abs(i - T(m_board_size(0))));*/
      T jerror; /*= min( abs(j-T(0)) , abs(j - T(m_board_size(1))));*/

      if (abs(i - T(0)) < abs(i - T(m_board_size(0))))
        ierror = abs(i - T(0));
      else
        ierror = abs(i - T(m_board_size(0)));

      if (abs(j - T(0)) < abs(j - T(m_board_size(1))))
        jerror = abs(j - T(0));
      else
        jerror = abs(j - T(m_board_size(1)));

      residuals[0] = ierror + jerror;
    } else {
      residuals[0] = T(0);
    }

    return true;
  }

private:
  Eigen::Vector2i m_board_size; // width height
  bool m_topleftWhite;
  double m_grid_length;
  bool m_laser_white;

  bool m_useOutofBoard;

  Eigen::Vector2d m_laserPoint;

  // square root of precision matrix
  Eigen::Matrix2d m_sqrtPrecisionMat; // 误差项的权重
};

class Pose3d2dError {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Pose3d2dError(cv::Point3d pt3d, cv::Point2d pt2d, Eigen::Vector4d camera)
      : pt3d_(pt3d), pt2d_(pt2d), camera_(camera) {}

  Pose3d2dError(Eigen::Vector3d pt3d, cv::Point2d pt2d, Eigen::Vector4d camera)
      : pt2d_(pt2d), camera_(camera) {
    pt3d_.x = pt3d(0);
    pt3d_.y = pt3d(1);
    pt3d_.z = pt3d(2);
  }

  Pose3d2dError(cv::Point3d pt3d, cv::Point2d pt2d, cv::Mat camK)
      : pt3d_(pt3d), pt2d_(pt2d) {
    camera_(0) = camK.at<double>(0, 0);
    camera_(1) = camK.at<double>(0, 2);
    camera_(2) = camK.at<double>(1, 1);
    camera_(3) = camK.at<double>(1, 2);
  }

  template <typename T>
  bool operator()(const T *const r, // 旋转向量 3d
                  const T *const t, // 平移向量 3d
                  T *residuals) const {
    T predicted_pt[3]; //预测值
    T pt3[3];
    pt3[0] = T(pt3d_.x);
    pt3[1] = T(pt3d_.y);
    pt3[2] = T(pt3d_.z);
    ceres::AngleAxisRotatePoint(r, pt3, predicted_pt);
    predicted_pt[0] += t[0];
    predicted_pt[1] += t[1];
    predicted_pt[2] += t[2];

    predicted_pt[0] = predicted_pt[0] / predicted_pt[2];
    predicted_pt[1] = predicted_pt[1] / predicted_pt[2];

    //像素点转到归一化坐标
    //        predicted_pt[0] = T(camera_->fx_)*predicted_pt[0] +
    //        T(camera_->cx_); predicted_pt[1] = T(camera_->fy_)*predicted_pt[1]
    //        + T(camera_->cy_);
    predicted_pt[0] = T(camera_[0]) * predicted_pt[0] + T(camera_[1]);
    predicted_pt[1] = T(camera_[2]) * predicted_pt[1] + T(camera_[3]);

    //预测 - 观测
    residuals[0] = T(pt2d_.x) - predicted_pt[0];
    residuals[1] = T(pt2d_.y) - predicted_pt[1];

    //        cout << "2residuals: " << residuals[0] << " " << residuals[1] <<
    //        endl;

    // cout << "residuals:"<<residuals[0]+residuals[1]<<endl;
    return true;
  }

private:
  cv::Point3d pt3d_;
  cv::Point2d pt2d_;
  Eigen::Vector4d camera_; // 分别是fx,cx,fy,cy
};

class Optimization {
public:
  Optimization();

  Eigen::Isometry3d solvePose3d2dError(std::vector<Eigen::Vector3d> pts3d,
                                       std::vector<cv::Point2d> pts2d,
                                       cv::Mat K, Eigen::Vector3d &r_ceres,
                                       Eigen::Vector3d &t_ceres);

  void get_theta_t(VPointCloud::Ptr cloud, Eigen::Vector2i board_size,
                   Eigen::Vector2d gray_zone, bool topleftWhite,
                   double grid_length, Eigen::Vector3d &theta_t,
                   bool useOutofBoard = true);

  void get_board_corner(cv::Size boardSize, double squareSize,
                        std::vector<cv::Point3d> &bBoardCorner);
  Eigen::Isometry3d solvePnP(std::vector<cv::Point2f> corners, cv::Mat camK,
                             cv::Size boardSize, double squareSize);
};

#endif // OPTIMIZATION_H


/*
 * @Author: huangwei barry.huangw@gmail.com
 * @Date: 2024-10-15 16:29:18
 * @LastEditors: huangwei barry.huangw@gmail.com
 * @LastEditTime: 2024-10-21 17:04:44
 * @FilePath: /livox_cam_calib/src/include/types.h
 */
#ifndef TYPES_H_
#define TYPES_H_

#include <pcl/common/common.h>

typedef pcl::PointXYZI VPoint;
typedef pcl::PointCloud<VPoint> VPointCloud;
typedef pcl::PointXYZRGB RGBPoint;
typedef pcl::PointCloud<RGBPoint> RGBPointCloud;

#endif


/*
 * @Author: huangwei barry.huangw@gmail.com
 * @Date: 2024-10-16 13:29:31
 * @LastEditors: huangwei barry.huangw@gmail.com
 * @LastEditTime: 2024-10-21 17:25:06
 * @FilePath: /livox_cam_calib/src/include/Visualization.h
 * @Description:
 */
#ifndef VISUALIZATION_H_
#define VISUALIZATION_H_

#include <Eigen/Core>
#include <iostream>

#include <pcl/visualization/cloud_viewer.h> /// pcl::visualization::PCLVisualizer

#include "common.hpp"
#include "types.h"

class Visualization {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Visualization() : m_confirm_flag(false), m_reject_flag(false) {
    m_plane_index = 0;
    update_flag = true;

    top_white_change = true;
    inverse_change = false;
  }
  void init_viewer(std::string name = "viewer") {
    viewer = pcl::visualization::PCLVisualizer::Ptr(
        new pcl::visualization::PCLVisualizer(name));
    viewer->addCoordinateSystem(1.0);
    viewer->setCameraPosition(-12, 0, 5, 0, 0, 0, 0, 0, 0, 0);
    // viewer->setSize(1500, 1200);
  }

  void reset(void);

  void register_keyboard(std::string type) {
    if (type == "get_chessboard")
      viewer->registerKeyboardCallback(&Visualization::keyboard_get_chessboard,
                                       *this, 0);
    if (type == "get_corner")
      viewer->registerKeyboardCallback(&Visualization::keyboard_get_corner,
                                       *this, 0);
  }
  void keyboard_get_chessboard(const pcl::visualization::KeyboardEvent &event,
                               void *viewer_void);
  void keyboard_get_corner(const pcl::visualization::KeyboardEvent &event,
                           void *viewer_void);

  void add_color_cloud(VPointCloud::Ptr cloud, Eigen::Vector3i color,
                       std::string id, int size = 2);
  void add_rgb_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbcloud,
                     std::string id);
  void add_sphere_origin(void);

  void close_viewer(void);

  pcl::visualization::PCLVisualizer::Ptr viewer;

  bool m_confirm_flag;
  bool m_reject_flag;

  bool update_flag;
  unsigned int m_plane_index;

  bool inverse_change;
  bool top_white_change;

private:
  std::vector<pcl::PointIndices> cluster_indices;
  VPointCloud::Ptr m_cluster_cloud;
};

#endif //
