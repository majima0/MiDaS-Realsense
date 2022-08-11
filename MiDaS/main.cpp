#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "util.h"
#include <stdio.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/highgui/highgui_c.h>
#include "cv-helpers.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
/*
int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() )
    {
        return -1;
    }

    // Read Network
    //const std::string model = "../model-f6b98070.onnx"; // MiDaS v2.1 Large
    const std::string model = "../model-small.onnx"; // MiDaS v2.1 Small
    cv::dnn::Net net = cv::dnn::readNet( model );
    if( net.empty() )
    {
        return -1;
    }

    // Set Preferable Backend and Target
    //CPU
    //net.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    //net.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );
    //GPU
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    while( true ){
        // Read Frame
        cv::Mat input;
        capture >> input;
        if( input.empty() ){
            cv::waitKey( 0 );
            break;
        }
        if( input.channels() == 4 ){
            cv::cvtColor( input, input, cv::COLOR_BGRA2BGR );
        }

        // Create Blob from Input Image
        // MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        //cv::Mat blob = cv::dnn::blobFromImage( input, 1 / 255.f, cv::Size( 384, 384 ), cv::Scalar( 123.675, 116.28, 103.53 ), true, false );
        // MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        cv::Mat blob = cv::dnn::blobFromImage( input, 1 / 255.f, cv::Size( 256, 256 ), cv::Scalar( 123.675, 116.28, 103.53 ), true, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        cv::Mat output = net.forward( getOutputsNames( net )[0] );

        // Convert Size to 384x384 from 1x384x384
        const std::vector<int32_t> size = { output.size[1], output.size[2] };
        output = cv::Mat( static_cast< int32_t >( size.size() ), &size[0], CV_32F, output.ptr<float>() );

        // Resize Output Image to Input Image Size
        cv::resize( output, output, input.size() );

        // Visualize Output Image
        // 1. Normalize ( 0.0 - 1.0 )
        // 2. Scaling ( 0 - 255 )
        double min, max;
        cv::minMaxLoc( output, &min, &max );
        const double range = max - min;
        output.convertTo( output, CV_32F, 1.0 / range, -( min / range ) );
        output.convertTo( output, CV_8U, 255.0 );
        cv::Mat dist = ~output;
        // Show Image
        cv::imshow( "input", input );
        cv::imshow( "output", dist );

        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
*/

using namespace cv;
using namespace std;
using namespace rs2;
const size_t inWidth = 640;
const size_t inHeight = 480;
const float WHRatio = inWidth / (float)inHeight;

int main(int argc, char** argv) try
{
    // Read Network
    //const std::string model = "../model-f6b98070.onnx"; // MiDaS v2.1 Large
    const std::string model = "../model-small.onnx"; // MiDaS v2.1 Small
    cv::dnn::Net net = cv::dnn::readNet(model);
    if (net.empty())
    {
        return -1;
    }

    // Set Preferable Backend and Target
    //CPU
    //net.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );
    //net.setPreferableTarget( cv::dnn::DNN_TARGET_CPU );
    //GPU
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR)
        .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),
            profile.height());
    }
    else
    {
        cropSize = Size(profile.width(),
            static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
        (profile.height() - cropSize.height) / 2),
        cropSize);

    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    float pixel_distance_in_meters;
    while (cvGetWindowHandle(window_name))
    {
        auto start = std::chrono::system_clock::now();
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, 
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        auto depth_mat = depth_frame_to_meters(depth_frame);

        // Create Blob from Input Image
        // MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        //cv::Mat blob = cv::dnn::blobFromImage( input, 1 / 255.f, cv::Size( 384, 384 ), cv::Scalar( 123.675, 116.28, 103.53 ), true, false );
        // MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        cv::Mat blob = cv::dnn::blobFromImage(color_mat, 1 / 255.f, cv::Size(256, 256), cv::Scalar(123.675, 116.28, 103.53), true, false);

        // Set Input Blob
        net.setInput(blob);

        // Run Forward Network
        cv::Mat output = net.forward(getOutputsNames(net)[0]);

        // Convert Size to 384x384 from 1x384x384
        const std::vector<int32_t> size = { output.size[1], output.size[2] };
        output = cv::Mat(static_cast<int32_t>(size.size()), &size[0], CV_32F, output.ptr<float>());

        // Resize Output Image to Input Image Size
        cv::resize(output, output, color_mat.size());

        // Visualize Output Image
        // 1. Normalize ( 0.0 - 1.0 )
        // 2. Scaling ( 0 - 255 )
        double min, max;
        cv::minMaxLoc(output, &min, &max);
        const double range = max - min;
        output.convertTo(output, CV_32F, 1.0 / range, -(min / range));
        output.convertTo(output, CV_8U, 255.0);
        cv::Mat dist = ~output;
        // Show Image
        cv::imshow("output", dist);

        //get depth
        //画像の真ん中のxy座標の距離を取得
        pixel_distance_in_meters = depth_frame.get_distance(inWidth / 2, inHeight / 2);
        std::cout << pixel_distance_in_meters << std::endl;

        imshow(window_name, color_mat);
        imshow("img", depth_mat);
        
        const int32_t key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
        auto end = std::chrono::system_clock::now();
        auto dur = end - start;        // 要した時間を計算
        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        // 要した時間をミリ秒（1/1000秒）に変換して表示
        std::cout << msec << " milli sec \n";
    }
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}