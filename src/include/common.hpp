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
