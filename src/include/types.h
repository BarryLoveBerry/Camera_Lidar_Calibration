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