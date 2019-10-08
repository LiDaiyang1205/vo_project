
#ifndef MAPPOINT_H
#define MAPPOINT_H
#include <myslam/common_include.h>
namespace myslam{
    class Frame;
    class MapPoint{
    public:
        typedef shared_ptr<MapPoint> Ptr;
        unsigned long id_;
        static unsigned long factory_id_;    // factory id
        bool        good_;      // 好的点
        Vector3d pos_;  // 世界坐标系中的位置
        Vector3d norm_; // 视线方向法线
        Mat descriptor_;  // 匹配的描述子
        list<Frame*>    observed_frames_;   // key-frames that can observe this point
        int         matched_times_;     // being an inliner in pose estimation
        int         visible_times_;     // being visible in current frame

        MapPoint();
        MapPoint(
                unsigned long id,
                const Vector3d& position,
                const Vector3d& norm,
                Frame* frame=nullptr,
                const Mat& descriptor=Mat());

        inline cv::Point3f getPositionCV() const {
            return cv::Point3f( pos_(0,0), pos_(1,0), pos_(2,0) );
        }

        static MapPoint::Ptr createMapPoint();
        static MapPoint::Ptr createMapPoint(
                const Vector3d& pos_world,
                const Vector3d& norm_,
                const Mat& descriptor,
                Frame* frame );
    };
}
#endif //MAPPOINT_H
