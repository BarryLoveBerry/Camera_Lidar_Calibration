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
