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
