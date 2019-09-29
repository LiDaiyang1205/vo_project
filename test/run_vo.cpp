// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main ( int argc, char** argv )
{
    //terminal运行时需要添加参数文件命令行，这里加一步判断
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    //链接参数文件
    myslam::Config::setParameterFile ( argv[1] );
    //构造VO，类型就是在VisualOdometry类中定义的指向自身类型的指针，然后用New开辟内存
    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );

    //读取数据文件夹地址
    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;

    //读取数据文件夹中的associate.txt文件
    ifstream fin ( dataset_dir+"/associate.txt" );
    //没读取成功的话输出错误
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    //定义图片名数组和时间戳数组，用于存放associate.txt文件中所示的时间戳对其的RGB图像和depth图像
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    //循环读取直到文件末尾
    while ( !fin.eof() )
    {
        //associate.txt文件中的数据肯定都是string类型的，定义按顺序从fin中输入。
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        //push_back进各个数组。
        //double atof (const char* str);
        //将一个单字节字符串转化成一个浮点数。
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        //.good()返回是否读取到文件末尾，文件末尾处此函数会返回false。所以跳出
        if ( fin.good() == false )
            break;
    }

    //创建相机
    myslam::Camera::Ptr camera ( new myslam::Camera );

    // visualization
    //可视化内容，用到OpenCV中的viz模块

    //第一步、创造一个可视化窗口，构造参数为窗口名称
    cv::viz::Viz3d vis("Visual Odometry");

    //第二步、创建坐标系部件，这里坐标系是以Widget部件类型存在的，
    // 构造参数是坐标系长度，也就是可视窗里的锥形小坐标系的长度，下面对坐标系部件进行设置
     cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    //这里设置坐标系部件属性，然后添加到视图窗口上去
    //首先利用setRenderingProperty()函数设置渲染属性，
    // 第一个参数是个枚举，对应要渲染的属性这里是线宽，后面是属性值
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    //用showWidget()函数将部件添加到窗口内
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Camera", camera_coor );
    //至此，窗口中已经显示了全部需要显示的东西，就是两个坐标系：世界坐标系，相机坐标系。
    //世界坐标系就是写死不动的了，所以后面也没有再提起过世界坐标系。需要做的就是计算出各个帧的相机坐标系位置
    //后续的核心就是下面的for循环，在循环中，不断的给相机坐标系设置新的pose，然后达到动画的效果。



    //第三步、设置视角。这步是非必要步骤，进行设置有利于观察，
    //不设置也会有默认视角，就是可能比较别扭。而且开始后拖动鼠标，也可以改变观察视角。
    //构建三个3D点,这里其实就是构造makeCameraPose()函数需要的三个向量：
    //相机位置坐标、相机焦点坐标、相机y轴朝向
    //蓝色-Z，红色-X，绿色-Y
    cv::Point3d cam_pos( 0, -1, -1 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    //由这三个参数，用makeCameraPose()函数构造Affine3d类型的相机位姿，这里其实是视角位姿，也就是程序开始时你处在什么视角看
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    //用setViewerPose()设置观看视角
    vis.setViewerPose( cam_pose );

    //输出RGB图像信息，共读到文件数
    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    //整个画面的快速刷新呈现动态，由此for循环控制。
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        //读取图像，创建帧操作
        Mat color = cv::imread ( rgb_files[i] );
        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        //这里加个每帧的运算时间，看看实时性
        boost::timer timer;
        //这里将帧添加进去，进行位姿变换计算
        vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed()<<endl;

        //VO状态为LOST时，跳出循环。
        if ( vo->state_ == myslam::VisualOdometry::LOST )
            break;
        //可视化窗口中动的是相机坐标系，所以本质上是求取相机坐标系下的点在世界坐标系下的坐标，
        //Pw=Twc*Pc;
        SE3 Twc = pFrame->T_c_w_.inverse();
        //SE3 Twc = pFrame->T_c_w_;


        //show the map and the camera pose
        //用Twc构造Affine3d类型的pose所需要的旋转矩阵和平移矩阵
        cv::Affine3d::Mat3 rmat(
                Twc.rotation_matrix()(0,0), Twc.rotation_matrix()(0,1), Twc.rotation_matrix()(0,2),
                Twc.rotation_matrix()(1,0), Twc.rotation_matrix()(1,1), Twc.rotation_matrix()(1,2),
                Twc.rotation_matrix()(2,0), Twc.rotation_matrix()(2,1), Twc.rotation_matrix()(2,2)
        );
        cv::Affine3d::Vec3 tvec(Twc.translation()(0,0), Twc.translation()(1,0), Twc.translation()(2,0));
        //构造位姿
        cv::Affine3d pose(rmat,tvec);

        //两窗口同时显示，一个是图像
        cv::imshow("image", color );
        cv::waitKey(1);
        //另外一个就是viz可视化窗口
        vis.setWidgetPose( "Camera", pose);
        vis.spinOnce(1, false);
    }

    return 0;
}
