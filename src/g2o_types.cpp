#include "myslam/g2o_types.h"

namespace myslam
{
//前两种这里没有用
//第三种，重投影误差
    void EdgeProjectXYZ2UVPoseOnly::computeError()
    {
        //顶点数组中取出顶点，转换成位姿指针类型，其实左边的pose类型可以写为auto
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        //误差计算，测量值减去估计值，也就是重投影误差
        //估计值计算方法是T*p,得到相机坐标系下坐标，然后在利用camera2pixel()函数得到像素坐标。
        _error = _measurement - camera_->camera2pixel(pose->estimate().map(point_) );
    }

    void EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
    {
        /**
         * 这里说一下整体思路：
         * 重投影误差的雅克比矩阵在书中P164页式7.45已经呈现，所以这里就是直接构造，
         * 构造时发现需要变换后的空间点坐标，所以需要先求出。
         */

        //首先还是从顶点取出位姿
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
        //这由位姿构造一个四元数形式T
        g2o::SE3Quat T ( pose->estimate() );
        //用T求得变换后的3D点坐标。T*p
        Vector3d xyz_trans = T.map ( point_ );
        //到这步，变换后的3D点xyz坐标就分别求出来了，后面的z平方，纯粹是为了后面构造J时方便定义的，因为需要多处用到
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z*z;

        //直接各个元素构造J就好了，对照式7.45是一模一样的，2*6的矩阵。
        _jacobianOplusXi ( 0,0 ) =  x*y/z_2 *camera_->fx_;
        _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x/z_2 ) ) *camera_->fx_;
        _jacobianOplusXi ( 0,2 ) = y/z * camera_->fx_;
        _jacobianOplusXi ( 0,3 ) = -1./z * camera_->fx_;
        _jacobianOplusXi ( 0,4 ) = 0;
        _jacobianOplusXi ( 0,5 ) = x/z_2 * camera_->fx_;

        _jacobianOplusXi ( 1,0 ) = ( 1+y*y/z_2 ) *camera_->fy_;
        _jacobianOplusXi ( 1,1 ) = -x*y/z_2 *camera_->fy_;
        _jacobianOplusXi ( 1,2 ) = -x/z *camera_->fy_;
        _jacobianOplusXi ( 1,3 ) = 0;
        _jacobianOplusXi ( 1,4 ) = -1./z *camera_->fy_;
        _jacobianOplusXi ( 1,5 ) = y/z_2 *camera_->fy_;
    }
}


