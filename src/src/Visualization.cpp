/*
 * @Author: huangwei barry.huangw@gmail.com
 * @Date: 2024-10-16 13:29:40
 * @LastEditors: huangwei barry.huangw@gmail.com
 * @LastEditTime: 2024-10-16 14:19:22
 * @FilePath: /livox_cam_calib/src/src/Visualization.cpp
 * @Description:
 */
// /*
//  * @Author: huangwei barry.huangw@gmail.com
//  * @Date: 2024-10-16 13:29:40
//  * @LastEditors: huangwei barry.huangw@gmail.com
//  * @LastEditTime: 2024-10-16 13:33:08
//  * @FilePath: /livox_cam_calib/src/src/Visualization.cpp
//  * @Description:
//  */
// #include "Visualization.h"

// void Visualization::reset() {
//   viewer->removeAllPointClouds();
//   m_plane_index = 0;
//   update_flag = true;

//   m_confirm_flag = false;
//   m_reject_flag = false;

//   inverse_change = true;
//   top_white_change = true;
// }

// void Visualization::keyboard_get_chessboard(
//     const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
//   if (event.getKeySym() == "w" && event.keyDown()) {
//     update_flag = true;
//     m_plane_index++;
//   }
//   if (event.getKeySym() == "s" && event.keyDown() && m_plane_index > 0) {
//     update_flag = true;
//     m_plane_index--;
//   }
//   if (event.getKeySym() == "o" && event.keyDown()) {
//     m_confirm_flag = true;
//   }
//   if (event.getKeySym() == "r" && event.keyDown()) {
//     m_confirm_flag = true;
//     m_reject_flag = true;
//   }
// }

// void Visualization::keyboard_get_corner(
//     const pcl::visualization::KeyboardEvent &event, void *viewer_void) {
//   if (event.getKeySym() == "k" && event.keyDown()) {
//     m_confirm_flag = true;
//   }
//   if (event.getKeySym() == "d" && event.keyDown()) {
//     update_flag = true;
//     top_white_change = true;
//   }
//   if (event.getKeySym() == "a" && event.keyDown()) {
//     update_flag = true;
//     inverse_change = true;
//   }
//   if (event.getKeySym() == "r" && event.keyDown()) {
//     m_confirm_flag = true;
//     m_reject_flag = true;
//   }
// }

// void Visualization::add_color_cloud(VPointCloud::Ptr cloud,
//                                     Eigen::Vector3i color, std::string id,
//                                     int size) {
//   pcl::visualization::PointCloudColorHandlerCustom<VPoint>
//   target_color_cloud(
//       cloud, color(0), color(1), color(2));
//   viewer->removePointCloud(id);
//   viewer->addPointCloud<VPoint>(cloud, target_color_cloud, id);
//   viewer->setPointCloudRenderingProperties(
//       pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, id);
// }

// void Visualization::add_rgb_cloud(
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbcloud, std::string id) {

//   pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>
//   rgb_view(
//       rgbcloud);
//   viewer->removePointCloud(id);
//   viewer->addPointCloud<pcl::PointXYZRGB>(rgbcloud, rgb_view, id);
//   viewer->setPointCloudRenderingProperties(
//       pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, id);
// }

// void Visualization::add_sphere_origin() {
//   pcl::ModelCoefficients sphere_coeff;
//   sphere_coeff.values.resize(4); // We need 4 values
//   sphere_coeff.values[0] = 0;
//   sphere_coeff.values[1] = 0;
//   sphere_coeff.values[2] = 0;
//   sphere_coeff.values[3] = 0.1;

//   viewer->addSphere(sphere_coeff);
// }

// void Visualization::close_viewer() {
//   // viewer->close();
//   viewer->~PCLVisualizer();
// }
