#ifndef FRAME_H
#define FRAME_H
#include <myslam/common_include.h>
#include "myslam/camera.h"
namespace myslam{
    class MapPoint;
class Frame{
public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long id_;  // frame 的 id
    double time_stamp_; // 什么时候呼叫,时间戳
    SE3 T_c_w_; // 世界坐标系到相机坐标系的变换
    Camera::Ptr camera_; // RGBD相机模式
    Mat color_, depth_; //彩色和深度图像

public:
    Frame();
    Frame( long id, double time_stamp=0, SE3 T_c_w=SE3(), Camera::Ptr camera=nullptr,
            Mat color=Mat(), Mat depth=Mat());
    ~Frame();

    // 工厂函数
    static Frame::Ptr createFrame();
    // 从深度图中获取指定点的深度信息
    double findDepth(const cv::KeyPoint& kp);
    // 获取相机光心
    Vector3d getCamCenter() const ;
    // 检查point是否在该frame中，在视野内
    bool isInFrame(const Vector3d& pt_world);
};
}

#endif //FRAME_H
