#include "myslam/frame.h"
#include "myslam/common_include.h"

namespace myslam
{
    Frame::Frame()
            : id_(-1), time_stamp_(-1), camera_(nullptr)
    {

    }

    Frame::Frame ( long id, double time_stamp, SE3 T_c_w, Camera::Ptr camera, Mat color, Mat depth )
            : id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth)
    {

    }

    Frame::~Frame()
    {

    }

    // 由factory_id++一个数去构造Frame对象时，调用的是默认构造函数，
    // 由于默认构造函数全都有默认值，所以就是按坑填，先填第一个id_，
    // 所以也就是相当于创建了一个只有ID号的空白帧。
    Frame::Ptr Frame::createFrame()
    {
        static long factory_id = 0;
        return Frame::Ptr( new Frame(factory_id++) );
    }

    double Frame::findDepth ( const cv::KeyPoint& kp )
    {
        int x = cvRound(kp.pt.x);
        int y = cvRound(kp.pt.y);
        ushort d = depth_.ptr<ushort>(y)[x];//这个是.ptr模板函数定位像素值的方法，记住用法
        if ( d!=0 )
        {
            return double(d)/camera_->depth_scale_;//除以比例尺
        }
        else
        {
            // check the nearby points
            int dx[4] = {-1,0,1,0};
            int dy[4] = {0,-1,0,1};
            for ( int i=0; i<4; i++ )
            {
                d = depth_.ptr<ushort>( y+dy[i] )[x+dx[i]];
                if ( d!=0 )
                {
                    return double(d)/camera_->depth_scale_;
                }
            }
        }
        return -1.0;
    }

    //获取相机光心。
    // 这里瞪大眼看！.translation()是取平移部分！不是取转置！
    // T_c_w_.inverse()求出来的平移部分就是R^(-1)*(-t),
    // 也就是相机坐标系下的(0,0,0)在世界坐标系下的坐标，
    // 也就是相机光心的世界坐标！
    Vector3d Frame::getCamCenter() const
    {
        return T_c_w_.inverse().translation();
    }

    bool Frame::isInFrame ( const Vector3d& pt_world )
    {
        Vector3d p_cam = camera_->world2camera( pt_world, T_c_w_ );
        if ( p_cam(2,0)<0 ) //z值
            return false;
        Vector2d pixel = camera_->world2pixel( pt_world, T_c_w_ );
        return pixel(0,0)>0 && pixel(1,0)>0 //xy值都大于0并且小于color图的行列
               && pixel(0,0)<color_.cols
               && pixel(1,0)<color_.rows;
    }

}

