#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <fstream>
#include <limits>

using namespace std;
using namespace cv;
using namespace optflow;

const String keys = "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1               }"
        "{@image2        |      | image2               }"
        "{@algorithm     |      | [farneback, simpleflow, tvl1, deepflow, sparsetodenseflow, RLOF_EPIC, RLOF_RIC, pcaflow, DISflow_ultrafast, DISflow_fast, DISflow_medium] }"
        "{@groundtruth   |      | path to the .flo file  (optional), Middlebury format }"
        "{m measure      |endpoint| error measure - [endpoint or angular] }"
        "{r region       |all   | region to compute stats about [all, discontinuities, untextured] }"
        "{d display      |      | display additional info images (pauses program execution) }"
        "{g gpu          |      | use OpenCL}"
        "{prior          |      | path to a prior file for PCAFlow}";

inline bool isFlowCorrect( const Point2f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9);
}
inline bool isFlowCorrect( const Point3f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && !cvIsNaN(u.z) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9)
            && (fabs(u.z) < 1e9);
}
static Mat endpointError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    Mat result(flow1.size(), CV_32FC1);
    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1 = flow1(i, j);
            const Point2f u2 = flow2(i, j);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
            {
                const Point2f diff = u1 - u2;
                result.at<float>(i, j) = sqrt((float)diff.ddot(diff)); //distance
            } else
                result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return result;
}
static Mat angularError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    Mat result(flow1.size(), CV_32FC1);

    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1_2d = flow1(i, j);
            const Point2f u2_2d = flow2(i, j);
            const Point3f u1(u1_2d.x, u1_2d.y, 1);
            const Point3f u2(u2_2d.x, u2_2d.y, 1);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
                result.at<float>(i, j) = acos((float)(u1.ddot(u2) / norm(u1) * norm(u2)));
            else
                result.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return result;
}
// what fraction of pixels have errors higher than given threshold?
static float stat_RX( Mat errors, float threshold, Mat mask )
{
    CV_Assert(errors.size() == mask.size());
    CV_Assert(mask.depth() == CV_8U);

    int count = 0, all = 0;
    for ( int i = 0; i < errors.rows; ++i )
    {
        for ( int j = 0; j < errors.cols; ++j )
        {
            if ( mask.at<char>(i, j) != 0 )
            {
                ++all;
                if ( errors.at<float>(i, j) > threshold )
                    ++count;
            }
        }
    }
    return (float)count / all;
}
static float stat_AX( Mat hist, int cutoff_count, float max_value )
{
    int counter = 0;
    int bin = 0;
    int bin_count = hist.rows;
    while ( bin < bin_count && counter < cutoff_count )
    {
        counter += (int) hist.at<float>(bin, 0);
        ++bin;
    }
    return (float) bin / bin_count * max_value;
}
static void calculateStats( Mat errors, Mat mask = Mat(), bool display_images = false )
{
    float R_thresholds[] = { 0.5f, 1.f, 2.f, 5.f, 10.f };
    float A_thresholds[] = { 0.5f, 0.75f, 0.95f };
    if ( mask.empty() )
        mask = Mat::ones(errors.size(), CV_8U);
    CV_Assert(errors.size() == mask.size());
    CV_Assert(mask.depth() == CV_8U);

    //displaying the mask
    if(display_images)
    {
        namedWindow( "Region mask", WINDOW_AUTOSIZE );
        imshow( "Region mask", mask );
    }

    //mean and std computation
    Scalar s_mean, s_std;
    float mean, std;
    meanStdDev(errors, s_mean, s_std, mask);
    mean = (float)s_mean[0];
    std = (float)s_std[0];
    printf("Average: %.2f\nStandard deviation: %.2f\n", mean, std);

    //RX stats - displayed in percent
    float R;
    int R_thresholds_count = sizeof(R_thresholds) / sizeof(float);
    for ( int i = 0; i < R_thresholds_count; ++i )
    {
        R = stat_RX(errors, R_thresholds[i], mask);
        printf("R%.1f: %.2f%%\n", R_thresholds[i], R * 100);
    }

    //AX stats
    double max_value;
    minMaxLoc(errors, NULL, &max_value, NULL, NULL, mask);

    Mat hist;
    const int n_images = 1;
    const int channels[] = { 0 };
    const int n_dimensions = 1;
    const int hist_bins[] = { 1024 };
    const float iranges[] = { 0, (float) max_value };
    const float* ranges[] = { iranges };
    const bool uniform = true;
    const bool accumulate = false;
    calcHist(&errors, n_images, channels, mask, hist, n_dimensions, hist_bins, ranges, uniform,
            accumulate);
    int all_pixels = countNonZero(mask);
    int cutoff_count;
    float A;
    int A_thresholds_count = sizeof(A_thresholds) / sizeof(float);
    for ( int i = 0; i < A_thresholds_count; ++i )
    {
        cutoff_count = (int) (floor(A_thresholds[i] * all_pixels + 0.5f));
        A = stat_AX(hist, cutoff_count, (float) max_value);
        printf("A%.2f: %.2f\n", A_thresholds[i], A);
    }
}

static Mat flowToDisplay(const Mat flow)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}

int main( int argc, char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV optical flow evaluation app");
    if ( parser.has("help") || argc < 4 )
    {
        parser.printMessage();
        printf("EXAMPLES:\n");
        printf("./example_optflow_optical_flow_evaluation im1.png im2.png farneback -d \n");
        printf("\t - compute flow field between im1 and im2 with farneback's method and display it");
        printf("./example_optflow_optical_flow_evaluation im1.png im2.png simpleflow groundtruth.flo \n");
        printf("\t - compute error statistics given the groundtruth; all pixels, endpoint error measure");
        printf("./example_optflow_optical_flow_evaluation im1.png im2.png farneback groundtruth.flo -m=angular -r=untextured \n");
        printf("\t - as before, but with changed error measure and stats computed only about \"untextured\" areas");
        printf("\n\n Flow file format description: http://vision.middlebury.edu/flow/code/flow-code/README.txt\n\n");
        return 0;
    }
    String i1_path = parser.get<String>(0);
    String i2_path = parser.get<String>(1);
    String method = parser.get<String>(2);
    String groundtruth_path = parser.get<String>(3);
    String error_measure = parser.get<String>("measure");
    String region = parser.get<String>("region");
    bool display_images = parser.has("display");
    const bool useGpu = parser.has("gpu");

    if ( !parser.check() )
    {
        parser.printErrors();
        return 0;
    }

    cv::ocl::setUseOpenCL(useGpu);
    printf("OpenCL Enabled: %u\n", useGpu && cv::ocl::haveOpenCL());

    Mat i1, i2;
    Mat_<Point2f> flow, ground_truth;
    Mat computed_errors;
    i1 = imread(i1_path, 1);
    i2 = imread(i2_path, 1);

    if ( !i1.data || !i2.data )
    {
        printf("No image data \n");
        return -1;
    }
    if ( i1.size() != i2.size() || i1.channels() != i2.channels() )
    {
        printf("Dimension mismatch between input images\n");
        return -1;
    }
    // 8-bit images expected by all algorithms
    if ( i1.depth() != CV_8U )
        i1.convertTo(i1, CV_8U);
    if ( i2.depth() != CV_8U )
        i2.convertTo(i2, CV_8U);

    if ( (method == "farneback" || method == "tvl1" || method == "deepflow" || method == "DISflow_ultrafast" || method == "DISflow_fast" || method == "DISflow_medium") && i1.channels() == 3 )
    {   // 1-channel images are expected
        cvtColor(i1, i1, COLOR_BGR2GRAY);
        cvtColor(i2, i2, COLOR_BGR2GRAY);
    } else if ( method == "simpleflow" && i1.channels() == 1 )
    {   // 3-channel images expected
        cvtColor(i1, i1, COLOR_GRAY2BGR);
        cvtColor(i2, i2, COLOR_GRAY2BGR);
    }

    flow = Mat(i1.size[0], i1.size[1], CV_32FC2);
    Ptr<DenseOpticalFlow> algorithm;

    if ( method == "farneback" )
        algorithm = createOptFlow_Farneback();
    else if ( method == "simpleflow" )
        algorithm = createOptFlow_SimpleFlow();
    else if ( method == "tvl1" )
        algorithm = createOptFlow_DualTVL1();
    else if ( method == "deepflow" )
        algorithm = createOptFlow_DeepFlow();
    else if ( method == "sparsetodenseflow" )
        algorithm = createOptFlow_SparseToDense();
    else if (method == "RLOF_EPIC")
    {
        algorithm = createOptFlow_DenseRLOF();
        Ptr<DenseRLOFOpticalFlow> rlof = algorithm.dynamicCast< DenseRLOFOpticalFlow>();
        rlof->setInterpolation(INTERP_EPIC);
        rlof->setForwardBackward(1.f);
    }
    else if (method == "RLOF_RIC")
    {
        algorithm = createOptFlow_DenseRLOF();
        Ptr<DenseRLOFOpticalFlow> rlof = algorithm.dynamicCast< DenseRLOFOpticalFlow>();;
        rlof->setInterpolation(INTERP_RIC);
        rlof->setForwardBackward(1.f);
    }
    else if ( method == "pcaflow" ) {
        if ( parser.has("prior") ) {
            String prior = parser.get<String>("prior");
            printf("Using prior file: %s\n", prior.c_str());
            algorithm = makePtr<OpticalFlowPCAFlow>(makePtr<PCAPrior>(prior.c_str()));
        }
        else
            algorithm = createOptFlow_PCAFlow();
    }
    else if ( method == "DISflow_ultrafast" )
        algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_ULTRAFAST);
    else if (method == "DISflow_fast")
        algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);
    else if (method == "DISflow_medium")
        algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    else
    {
        printf("Wrong method!\n");
        parser.printMessage();
        return -1;
    }

    double startTick, time;
    startTick = (double) getTickCount(); // measure time

    if (useGpu)
        algorithm->calc(i1, i2, flow.getUMat(ACCESS_RW));
    else
        algorithm->calc(i1, i2, flow);

    time = ((double) getTickCount() - startTick) / getTickFrequency();
    printf("\nTime [s]: %.3f\n", time);
    if(display_images)
    {
        Mat flow_image = flowToDisplay(flow);
        namedWindow( "Computed flow", WINDOW_AUTOSIZE );
        imshow( "Computed flow", flow_image );
    }

    if ( !groundtruth_path.empty() )
    { // compare to ground truth
        ground_truth = readOpticalFlow(groundtruth_path);
        if ( flow.size() != ground_truth.size() || flow.channels() != 2
                || ground_truth.channels() != 2 )
        {
            printf("Dimension mismatch between the computed flow and the provided ground truth\n");
            return -1;
        }
        if ( error_measure == "endpoint" )
            computed_errors = endpointError(flow, ground_truth);
        else if ( error_measure == "angular" )
            computed_errors = angularError(flow, ground_truth);
        else
        {
            printf("Invalid error measure! Available options: endpoint, angular\n");
            return -1;
        }

        Mat mask;
        if( region == "all" )
            mask = Mat::ones(ground_truth.size(), CV_8U) * 255;
        else if ( region == "discontinuities" )
        {
            Mat truth_merged, grad_x, grad_y, gradient;
            vector<Mat> truth_split;
            split(ground_truth, truth_split);
            truth_merged = truth_split[0] + truth_split[1];

            Sobel( truth_merged, grad_x, CV_16S, 1, 0, -1, 1, 0, BORDER_REPLICATE );
            grad_x = abs(grad_x);
            Sobel( truth_merged, grad_y, CV_16S, 0, 1, 1, 1, 0, BORDER_REPLICATE );
            grad_y = abs(grad_y);
            addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); //approximation!

            Scalar s_mean;
            s_mean = mean(gradient);
            double threshold = s_mean[0]; // threshold value arbitrary
            mask = gradient > threshold;
            dilate(mask, mask, Mat::ones(9, 9, CV_8U));
        }
        else if ( region == "untextured" )
        {
            Mat i1_grayscale, grad_x, grad_y, gradient;
            if( i1.channels() == 3 )
                cvtColor(i1, i1_grayscale, COLOR_BGR2GRAY);
            else
                i1_grayscale = i1;
            Sobel( i1_grayscale, grad_x, CV_16S, 1, 0, 7 );
            grad_x = abs(grad_x);
            Sobel( i1_grayscale, grad_y, CV_16S, 0, 1, 7 );
            grad_y = abs(grad_y);
            addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); //approximation!
            GaussianBlur(gradient, gradient, Size(5,5), 1, 1);

            Scalar s_mean;
            s_mean = mean(gradient);
            // arbitrary threshold value used - could be determined statistically from the image?
            double threshold = 1000;
            mask = gradient < threshold;
            dilate(mask, mask, Mat::ones(3, 3, CV_8U));
        }

        else
        {
            printf("Invalid region selected! Available options: all, discontinuities, untextured");
            return -1;
        }

        //masking out NaNs and incorrect GT values
        Mat truth_split[2];
        split(ground_truth, truth_split);
        Mat abs_mask = Mat((abs(truth_split[0]) < 1e9) & (abs(truth_split[1]) < 1e9));
        Mat nan_mask = Mat((truth_split[0]==truth_split[0]) & (truth_split[1] == truth_split[1]));
        bitwise_and(abs_mask, nan_mask, nan_mask);

        bitwise_and(nan_mask, mask, mask); //including the selected region

        if(display_images) // display difference between computed and GT flow
        {
            Mat difference = ground_truth - flow;
            Mat masked_difference;
            difference.copyTo(masked_difference, mask);
            Mat flow_image = flowToDisplay(masked_difference);
            namedWindow( "Error map", WINDOW_AUTOSIZE );
            imshow( "Error map", flow_image );
        }

        printf("Using %s error measure\n", error_measure.c_str());
        calculateStats(computed_errors, mask, display_images);

    }
    if(display_images) // wait for the user to see all the images
        waitKey(0);
    return 0;

}
