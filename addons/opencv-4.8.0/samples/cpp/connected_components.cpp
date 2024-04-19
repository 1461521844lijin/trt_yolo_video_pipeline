
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat img;
int threshval = 100;

static void on_trackbar(int, void*)
{
    Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    Mat labelImage(img.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    Mat dst(img.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
         }
     }

    imshow( "Connected Components", dst );
}

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, "{@image|stuff.jpg|image for converting to a grayscale}");
    parser.about("\nThis program demonstrates connected components and use of the trackbar\n");
    parser.printMessage();
    cout << "\nThe image is converted to grayscale and displayed, another image has a trackbar\n"
            "that controls thresholding and thereby the extracted contours which are drawn in color\n";

    String inputImage = parser.get<string>(0);
    img = imread(samples::findFile(inputImage), IMREAD_GRAYSCALE);

    if(img.empty())
    {
        cout << "Could not read input image file: " << inputImage << endl;
        return EXIT_FAILURE;
    }

    imshow( "Image", img );

    namedWindow( "Connected Components", WINDOW_AUTOSIZE);
    createTrackbar( "Threshold", "Connected Components", &threshval, 255, on_trackbar );
    on_trackbar(threshval, 0);

    waitKey(0);
    return EXIT_SUCCESS;
}
