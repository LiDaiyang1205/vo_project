
#ifndef MAPPOINT_H
#define MAPPOINT_H
#include <myslam/common_include.h>
namespace myslam{
    class Frame;
    class MapPoint{
    public:
        typedef shared_ptr<MapPoint> Ptr;
        unsigned long id_;
        Vector3d pos_;  // 世界坐标系中的位置
        Vector3d norm_; // 视线方向法线
        Mat descriptor_;  // 匹配的描述子
        int observed_times_; // 被特征匹配观测的次数
        int correct_times_;  // 位姿估计匹配次数，匹配正确

        MapPoint();
        MapPoint(long id, Vector3d position, Vector3d norm);

        static MapPoint::Ptr createMapPoint();
    };
}
#endif //MAPPOINT_H
