#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{
    //默认构造函数，提供默认值、读取配置参数
    VisualOdometry::VisualOdometry():
            state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
    {
        num_of_features_    = Config::get<int> ( "number_of_features" );
        scale_factor_       = Config::get<double> ( "scale_factor" );
        level_pyramid_      = Config::get<int> ( "level_pyramid" );
        match_ratio_        = Config::get<float> ( "match_ratio" );
        max_num_lost_       = Config::get<float> ( "max_num_lost" );
        min_inliers_        = Config::get<int> ( "min_inliers" );
        key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
        key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
        map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
        //这个create()，之前用的时候，都是用的默认值，所以没有任何参数，这里传入了一些参数，可参见函数定义
        orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
    }

    VisualOdometry::~VisualOdometry()
    {
    }

    //最核心的添加帧，参数即为新的一帧，根据VO当前状态选择是进行初始化还是计算T
    bool VisualOdometry::addFrame ( Frame::Ptr frame )
    {
        //根据VO状态来进行不同处理。
        switch ( state_ )
        {
            //第一帧，则进行初始化处理
            case INITIALIZING:
            {
                //更改状态为OK
                state_ = OK;
                //因为是初始化，所以当前帧和参考帧都为此第一帧
                curr_ = ref_ = frame;
                //并将此帧插入到地图中
                // map_->insertKeyFrame ( frame );
                // extract features from first frame
                //匹配的操作，提取keypoint和计算描述子
                extractKeyPoints();
                computeDescriptors();
                // compute the 3d position of features in ref frame
                //这里提取出的keypoint要形成3d坐标，所以调用setRef3DPoints()去补齐keypoint的depth数据。
                //setRef3DPoints();
                //之前增加关键帧需调用map类中的insertKeyFrame()函数，
                // 这里第一帧的话，就直接调用自身的addKeyFrame()函数添加进地图
                addKeyFrame();      // the first frame is a key-frame

                break;
            }
            //如果为正常，则匹配并调用poseEstimationPnP()函数计算T。
            case OK:
            {
                //整个流程的改变就是只需要不断进行每一帧的位姿迭代，
                //而不需要用到参考帧的3D点进行匹配得T了
                curr_ = frame;
                //新的帧来了，先将其位姿赋值为参考帧的位姿，
                //因为考虑到匹配失败的情况下，这一帧就定义为丢失了，所以位姿就用参考帧的了。
                //如果一切正常，求得了当前帧的位姿，就进行赋值覆盖掉就好了。
                curr_->T_c_w_ = ref_->T_c_w_;
                extractKeyPoints();
                computeDescriptors();
                featureMatching();
                //进行位姿估计
                poseEstimationPnP();

                //根据位姿估计的结果进行分别处理
                if ( checkEstimatedPose() == true ) // a good estimation
                {
                    //正常求得位姿T，对当前帧位姿进行更新
                    curr_->T_c_w_ = T_c_w_estimated_;
                    optimizeMap();
                    num_lost_ = 0;
                    //检验一下是否为关键帧，是的话加入关键帧
                    if ( checkKeyFrame() == true ) // is a key-frame
                    {
                        addKeyFrame();
                    }
                }
                else // bad estimation due to various reasons
                {
                    //坏的估计将丢失计数+1，并判断是否大于最大丢失数，如果是，将VO状态切换为lost。
                    num_lost_++;
                    if ( num_lost_ > max_num_lost_ )
                    {
                        state_ = LOST;
                    }
                    return false;
                }
                break;
            }
            case LOST:
            {
                cout<<"vo has lost."<<endl;
                break;
            }
        }

        return true;
    }

    //提取keypoint
    void VisualOdometry::extractKeyPoints()
    {
        boost::timer timer;
        orb_->detect ( curr_->color_, keypoints_curr_ );
        cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
    }

    //计算描述子
    void VisualOdometry::computeDescriptors()
    {
        boost::timer timer;
        orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
        cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
    }

    //特征匹配
    void VisualOdometry::featureMatching()
    {
        boost::timer timer;
        vector<cv::DMatch> matches;
        // select the candidates in map
        //建立一个目标图，承接匹配需要地图点的描述子，因为匹配是需要的参数是描述子
        Mat desp_map;
        //建立一个候选地图点数组，承接匹配需要的地图点
        vector<MapPoint::Ptr> candidate;
        //检查地图点是否为匹配需要的，逻辑就是遍历维护的局部地图中所有地图点，
        //然后利用isInFrame()函数检查有哪些地图点在当前观察帧中，
        //如果在则把地图点push进candidate中，描述子push进desp_map中
        for ( auto& allpoints: map_->map_points_ )
        {
            //这里还是STL用法，allpoints为map类中定义的双模板类型类成员，此表示第二个模板类型
            //总之功能上就是把地图点取出，赋值给p
            MapPoint::Ptr& p = allpoints.second;
            // check if p in curr frame image
            //利用是否在匹配帧中来判断是否添加进去
            if ( curr_->isInFrame(p->pos_) )
            {
                // add to candidate
                //观察次数增加一次
                p->visible_times_++;
                //点push进candidate
                candidate.push_back( p );
                //描述子push进desp_map
                desp_map.push_back( p->descriptor_ );
            }
        }

        //这步匹配中，由原来的参考帧，变成了上面定义的desp_map地图，进行匹配。
        matcher_flann_.match ( desp_map, descriptors_curr_, matches );
        // select the best matches
        //寻找最小距离，这里用到了STL中的std::min_element和lambda表达式
        //这的作用是找到matches数组中最小的距离，然后赋值给min_dis
        float min_dis = std::min_element (
                matches.begin(), matches.end(),
                [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
                {
                    return m1.distance < m2.distance;
                } )->distance;

        match_3dpts_.clear();
        match_2dkp_index_.clear();
        for ( cv::DMatch& m : matches )
        {
            //根据最小距离，对matches数组进行刷选，只有小于最小距离一定倍率或者小于30的才能push_back进数组。
            //最终得到筛选过的，距离控制在一定范围内的可靠匹配
            if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
            {
                //这里变化是不像之前直接将好的m  push进feature_matches_就完了。
                //这里感觉像做一个记录，candidate中存的是观察到的地图点
                // 进一步，在candidate中选出好的匹配的点，push进match_3dpts_，
                //这个match_3dpts_代表当前这一帧计算T时所利用到的所有好的地图点，放入其中。
                //由此可见，candidate只是个中间存储，新的一帧过来会被刷新。
                //同样match_2dkp_index_也是同样的道理，只不过是记录的当前帧detect出来的keypoint数组中的点的索引。
                match_3dpts_.push_back( candidate[m.queryIdx] );
                match_2dkp_index_.push_back( m.trainIdx );
            }
        }
        cout<<"good matches: "<<match_3dpts_.size() <<endl;
        cout<<"match cost time: "<<timer.elapsed() <<endl;
    }

//    //新的帧来的时候，是一个2D数据，因为PNP需要的是参考帧的3D，当前帧的2D。
//    //所以在当前帧迭代为参考帧时，有个工作就是加上depth数据。也就是设置参考帧的3D点。
//    void VisualOdometry::setRef3DPoints()
//    {
//        // select the features with depth measurements
//        //3D点数组先清空，后面重新装入
//        pts_3d_ref_.clear();
//        //参考帧的描述子也是构建个空Mat。
//        descriptors_ref_ = Mat();
//        //对当前keypoints数组进行遍历
//        for ( size_t i=0; i<keypoints_curr_.size(); i++ )
//        {
//            //找到对应的depth数据赋值给d
//            double d = ref_->findDepth(keypoints_curr_[i]);
//            //如果>0说明depth数据正确，进行构造
//            if ( d > 0)
//            {
//                //由像素坐标求得相机下3D坐标
//                Vector3d p_cam = ref_->camera_->pixel2camera(Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d);
//                //由于列向量，所以按行构造Point3f，push_back进参考帧的3D点。
//                pts_3d_ref_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
//                //参考帧描述子这里就按照当前帧描述子按行push_back。这里也可以发现，算出来的Mat类型的描述子，是按行存储为一列，读取时需要遍历行。
//                descriptors_ref_.push_back(descriptors_curr_.row(i));
//            }
//        }
//    }

    //核心功能函数，用PnP估计位姿
    void VisualOdometry::poseEstimationPnP()
    {
        // construct the 3d 2d observations
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;

        //从这里就可以看出，地图点用的是3D，当前帧用的2D。
        for ( int index:match_2dkp_index_ )
        {
            pts2d.push_back ( keypoints_curr_[index].pt );
        }
        for ( MapPoint::Ptr pt:match_3dpts_ )
        {
            pts3d.push_back( pt->getPositionCV() );
        }


        //构造相机内参矩阵K
        Mat K = ( cv::Mat_<double>(3,3)<<
                                       ref_->camera_->fx_, 0, ref_->camera_->cx_,
                0, ref_->camera_->fy_, ref_->camera_->cy_,
                0,0,1
        );

        //旋转向量，平移向量，内点数组
        Mat rvec, tvec, inliers;
        //整个核心就是用这个cv::solvePnPRansac()去求解两帧之间的位姿变化
        cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
        //内点数量为内点行数，所以为列存储。
        num_inliers_ = inliers.rows;
        cout<<"pnp inliers: "<<num_inliers_<<endl;
        //根据旋转和平移构造出当前帧相对于参考帧的T，这样一个T计算完成了。循环计算就能得到轨迹。
        T_c_w_estimated_ = SE3(
                SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
                Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
        );

        // using bundle adjustment to optimize the pose
        //初始化，注意由于更新所需要的unique指针问题。
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
        Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        Block* solver_ptr = new Block( linearSolver );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm ( solver );

        //添加顶点，一帧只有一个位姿，也就是只有一个顶点
        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId ( 0 );
        pose->setEstimate ( g2o::SE3Quat (
                T_c_w_estimated_.rotation_matrix(),
                T_c_w_estimated_.translation()
        ) );
        optimizer.addVertex ( pose );

        // edges边有许多，每个特征点都对应一个重投影误差，也就有一个边。
        for ( int i=0; i<inliers.rows; i++ )
        {
            int index = inliers.at<int>(i,0);
            // 3D -> 2D projection
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0, pose);
            edge->camera_ = curr_->camera_.get();
            edge->point_ = Vector3d( pts3d[index].x, pts3d[index].y, pts3d[index].z );
            edge->setMeasurement( Vector2d(pts2d[index].x, pts2d[index].y) );
            edge->setInformation( Eigen::Matrix2d::Identity() );
            optimizer.addEdge( edge );
            // set the inlier map points
            match_3dpts_[index]->matched_times_++;
        }

        //开始优化
        optimizer.initializeOptimization();
        //设置迭代次数
        optimizer.optimize(10);

        //这步就是将优化后的结果，赋值给T_c_r_estimated_
        T_c_w_estimated_ = SE3 (
                pose->estimate().rotation(),
                pose->estimate().translation()
        );
        cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;

    }

    //简单的位姿检验函数，整体思路就是匹配点不能过少，运动不能过大。
    bool VisualOdometry::checkEstimatedPose()
    {
        // check if the estimated pose is good
        //这里简单的做一下位姿估计判断，主要有两个，一就是匹配点太少的话，直接false，或者变换向量模长太大的话，也直接false
        if ( num_inliers_ < min_inliers_ )
        {
            cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
            return false;
        }
        // if the motion is too large, it is probably wrong
        //将变换矩阵取log操作得到变换向量。
        SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        //根据变换向量的模长来判断运动的大小。过大的话返回false
        if ( d.norm() > 5.0 )
        {
            cout<<"reject because motion is too large: "<<d.norm()<<endl;
            return false;
        }
        //如果让面两项都没return，说明内点量不少，运动也没过大，return true
        return true;
    }

    bool VisualOdometry::checkKeyFrame()
    {
        //说一下这个是否为关键帧的判断，也很简单，
        //关键帧并不是之前理解的轨迹比较长了，隔一段选取一个，而还是每一帧的T都判断一下，比较大就说明为关键帧，说明在这一帧中，要么平移比较大，要么拐弯导致旋转比较大，所以添加，如果在运动上一直就是小运动，运动多久都不会添加为关键帧。
        //另外上方的判断T计算错误也是运动过大，但是量级不一样，判断计算错误是要大于5，而关键帧，在配置文件中看只需要0.1就定义为关键帧了，所以0.1到5的差距，导致这两个函数并不冲突
        SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
            return true;
        return false;
    }

//增加关键帧函数多了一步在第一帧时，将其对应的地图点全部添加进地图中。
    void VisualOdometry::addKeyFrame()
    {
        if ( map_->keyframes_.empty() )
        {
            // first key-frame, add all 3d points into map
            for ( size_t i=0; i<keypoints_curr_.size(); i++ )
            {
                double d = curr_->findDepth ( keypoints_curr_[i] );
                if ( d < 0 )
                    continue;
                Vector3d p_world = ref_->camera_->pixel2world (
                        Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_c_w_, d
                );
                Vector3d n = p_world - ref_->getCamCenter();
                n.normalize();
                //上方求出构造地图点所需所有参数，3D点、模长、描述子、帧，然后构造一个地图点
                MapPoint::Ptr map_point = MapPoint::createMapPoint(
                        p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
                );
                //添加进地图
                map_->insertMapPoint( map_point );
            }
        }
        //一样的，第一帧添加进关键帧
        map_->insertKeyFrame ( curr_ );
        ref_ = curr_;

    }
    //新增函数，增加地图中的点。随时的增删地图中的点，来跟随运动
    void VisualOdometry::addMapPoints()
    {
        // add the new map points into map
        //创建一个bool型的数组matched，大小为当前keypoints数组大小，值全为false
        vector<bool> matched(keypoints_curr_.size(), false);
        //首先这个match_2dkp_index_是新来的当前帧跟地图匹配时，好的匹配到的关键点在keypoins数组中的索引
        //在这里将已经匹配的keypoint索引置为true
        for ( int index:match_2dkp_index_ )
            matched[index] = true;
        //遍历当前keypoints数组
        for ( int i=0; i<keypoints_curr_.size(); i++ )
        {
            //如果为true，说明在地图中找到了匹配，也就意味着地图中已经有这个点了。直接continue
            if ( matched[i] == true )
                continue;
            //如果没有continue的话，说明这个点在地图中没有找到匹配，认定为新的点，
            //下一步就是找到depth数据，构造3D点，然后构造地图点，添加进地图即可。
            double d = ref_->findDepth ( keypoints_curr_[i] );
            if ( d<0 )
                continue;
            Vector3d p_world = ref_->camera_->pixel2world (
                    Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ),
                    curr_->T_c_w_, d
            );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                    p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
            );
            map_->insertMapPoint( map_point );
        }
    }

//新增函数：优化地图。主要是为了维护地图的规模。删除一些地图点，在点过少时增加地图点等操作。
    void VisualOdometry::optimizeMap()
    {
        // remove the hardly seen and no visible points
        //删除地图点，遍历地图中的地图点。并分几种情况进行删除。
        for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
        {
            //如果点在当前帧都不可见了，说明跑的比较远，删掉
            if ( !curr_->isInFrame(iter->second->pos_) )
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            //定义匹配率，用匹配次数/可见次数，匹配率过低说明经常见但是没有几次匹配。应该是一些比较难识别的点，也就是出来的描述子比较奇葩。所以删掉
            float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
            if ( match_ratio < map_point_erase_ratio_ )
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }

            double angle = getViewAngle( curr_, iter->second );
            if ( angle > M_PI/6. )
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            //继续，可以根据一些其他条件自己添加要删除点的情况
            if ( iter->second->good_ == false )
            {
                // TODO try triangulate this map point
            }
            iter++;
        }

        //下面说一些增加点的情况，首先当前帧去地图中匹配时，点少于100个了，
        // 一般情况是运动幅度过大了，跟之前的帧没多少交集了，所以增加一下。
        if ( match_2dkp_index_.size()<100 )
            addMapPoints();
        //如果点过多了，多于1000个，适当增加释放率，让地图处于释放点的趋势。
        if ( map_->map_points_.size() > 1000 )
        {
            // TODO map is too large, remove some one
            map_point_erase_ratio_ += 0.05;
        }
            //如果没有多于1000个，保持释放率在0.1，维持地图的体量为平衡态
        else
            map_point_erase_ratio_ = 0.1;
        cout<<"map points: "<<map_->map_points_.size()<<endl;
    }

//取得一个空间点在一个帧下的视角角度。返回值是double类型的角度值。
    double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
    {
        //构造发方法是空间点坐标减去相机中心坐标。得到从相机中心指向指向空间点的向量。
        Vector3d n = point->pos_ - frame->getCamCenter();
        //单位化
        n.normalize();
        //返回一个角度，acos()为反余弦，
        //向量*乘为：a*b=|a||b|cos<a,b>
        //所以单位向量的*乘得到的是两个向量的余弦值，再用acos()即得到角度，返回
        //物理意义就是得到世界坐标系下看空间点和从相机坐标系下看空间点，视角差的角度。
        return acos( n.transpose()*point->norm_ );
    }

}
