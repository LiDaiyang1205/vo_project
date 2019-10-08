
#include "myslam/common_include.h"
#include "myslam/mappoint.h"

namespace myslam
{

    //默认构造函数，设定各种默认值
    MapPoint::MapPoint()
            : id_(-1), pos_(Vector3d(0,0,0)), norm_(Vector3d(0,0,0)),  good_(true), visible_times_(0), matched_times_(0)
    {

    }
    //构造函数，将观察帧push_back进去
    MapPoint::MapPoint ( long unsigned int id, const Vector3d& position, const Vector3d& norm, Frame* frame, const Mat& descriptor )
            : id_(id), pos_(position), norm_(norm), good_(true), visible_times_(1), matched_times_(1), descriptor_(descriptor)
    {
        observed_frames_.push_back(frame);
    }

    // 初始化factory id
    unsigned long MapPoint::factory_id_ = 0;

    //创建地图点时，直接在累加上ID然后构造一个就好了。返回定义的MapPoint类型指针Ptr
    MapPoint::Ptr MapPoint::createMapPoint()
    {
        return MapPoint::Ptr(new MapPoint(factory_id_++, Vector3d(0,0,0), Vector3d(0,0,0)));
    }

    MapPoint::Ptr MapPoint::createMapPoint (
            const Vector3d& pos_world,
            const Vector3d& norm,
            const Mat& descriptor,
            Frame* frame )
    {
        return MapPoint::Ptr(
                new MapPoint( factory_id_++, pos_world, norm, frame, descriptor )
        );
    }

}
