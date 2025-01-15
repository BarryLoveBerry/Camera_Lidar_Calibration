老铁们 觉得不错 请给个star 谢谢

## 电脑依赖环境
### 电脑需要提前ROS环境和确保下列几个的库都是安装好的
ROS
Eigen3
OpenCV4
PCL1.7
Boost
Ceres
Sophus
## 相机内参标定
先进行相机的内参标定，可以参考如下文档进行相机的内参标定
https://blog.csdn.net/wakeup_high/article/details/138030786

## 录取数据的topic
相机内参标定完成后，有了相机的内参就可以用联合标定工具进行相机和激光雷达的联合标定

## 标定数据要求
至少进行远，近，左，右，前方不同位置的标定板数据采集
录制话题   
lidar: 订阅自己的相关雷达话题
camera: 订阅自己的相关相机话题
（在launch文件中进行修改配置）

# 操作流程
## 配置参数
src/config/app.yaml: 此文件中主要涉及相机的内参，畸变信息，图像尺寸，标定板的尺寸（宽*高），网格大小
通过上一步完成的相机内参，进行填写
src/launch/colored_pointcloud_node161.launch: 节点启动配置文件

在这个文件中，主要检查Camera和lidar的topic是否要修改
标定结果数据路径需要配置result_save_file_path
标定数据的数量，bag_num默认6帧
## 编译启动
完成1中的参数配置后，进行程序编译
catkin_make install 
source install/setup.bash
roslaunch livox_cam_calib colored_pointcloud_node161.launch
启动后, 会弹出rviz界面

## 数据回放
进行数据回放
rosbag play -l xxxxx.bag
选取lidar和camera数据
在程序启动的terminal， 按n 进行点云和相机数据抓取

抓取成功后，终端会显示角点检测成功！
rviz会显示对应的点云

## 检查角点检测的精度
然后通过pulish point功能，进行标定板选取（中间位置），标定板的点云角点会自动计算出来
如下图

## 检查点云角点质量
通过查看相机角点检测位置和点云位置，筛选有质量的标定数据

## 记录标定数据
完成上一步的角点质量检测后，再程序运行终端按k，程序会记录下当前帧的数据，并且会把camera和lidar的角点检测信息，图像，保存下来
计算外参
重复步骤1-7
采集的数量达到配置的数量后，程序则自动利用标定的数据进行外参矩阵的解算
会弹出相应的投影图片，投影结果和外参也会保存到指定路径


TODO：
优化雷达角点检测算法
