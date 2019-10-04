
#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>
namespace myslam{
    class VisualOdometry{
    public:
        typedef shared_ptr<VisualOdometry> Ptr;
        //定义枚举来表征VO状态，分别为：初始化、正常、丢失
        enum VOState{
            INITIALIZING=-1,
            OK=0,
            LOST
        };

        //这里为两两帧VO所用到的参考帧和当前帧。还有VO状态和整个地图。
        VOState state_; // 当前VO状态
        Map::Ptr map_; // 有所有frame和map point的map
        Frame::Ptr ref_; // reference frame 参考帧
        Frame::Ptr curr_; // current frame 当前帧

        //这里是两帧匹配需要的：keypoints，descriptors，matches
        cv::Ptr<cv::ORB> orb_; // orb 检测与计算
        vector<cv::Point3f> pts_3d_ref_; // 参考帧的3d点
        vector<cv::KeyPoint> keypoints_curr_; // 当前帧的keypoint
        Mat descriptors_curr_; // 当前帧的描述子
        Mat descriptors_ref_; // 参考帧的描述子
        vector<cv::DMatch> feature_matches_; // 特征匹配

        //这里为匹配结果T，还有表征结果好坏的内点数和丢失数
        SE3 T_c_r_estimated_; // 当前帧估计位姿
        int num_inliers_; // 好的特征数量
        int num_lost_; // 丢失数

        // 参数
        int num_of_features_; // 特征数量
        double scale_factor_; // 图像金字塔的尺度因子
        int level_pyramid_; // 图像金字塔的层数
        float match_ratio_; // 选择好的匹配的系数
        int max_num_lost_; // 持续丢失时间的最大值
        int min_inliers_; // 最小 inliers

        //用于判定是否为关键帧的标准，就是规定一定幅度的旋转和平移，大于这个幅度就归为关键帧
        double key_frame_min_rot; // 两个关键帧的最小旋转
        double key_frame_min_trans; // 两个关键帧的最小平移

    public:// 公式
        VisualOdometry();
        ~VisualOdometry();

        //这个函数为核心处理函数，将帧添加进来，然后处理。
        bool addFrame(Frame::Ptr frame); // 增加新的帧

    protected:
        //一些内部处理函数，这块主要是特征匹配的
        void extractKeyPoints();
        void computeDescriptors();
        void featureMatching();
        void poseEstimationPnP();
        void setRef3DPoints();

        //这里是关键帧的一些功能函数
        void addKeyFrame();
        bool checkEstimatedPose();
        bool checkKeyFrame();

    };
}

#endif //VISUAL_ODOMETRY_H
