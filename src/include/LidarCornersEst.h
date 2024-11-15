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