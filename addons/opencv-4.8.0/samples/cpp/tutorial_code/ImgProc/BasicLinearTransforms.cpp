/**
 * @file BasicLinearTransforms.cpp
 * @brief Simple program to change contrast and brightness
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cin;
using std::cout;
using std::endl;
using namespace cv;

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
    /// Read image given by user
    //! [basic-linear-transform-load]
    CommandLineParser parser( argc, argv, "{@input | lena.jpg | input image}" );
    Mat image = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if( image.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }
    //! [basic-linear-transform-load]

    //! [basic-linear-transform-output]
    Mat new_image = Mat::zeros( image.size(), image.type() );
    //! [basic-linear-transform-output]

    //! [basic-linear-transform-parameters]
    double alpha = 1.0; /*< Simple contrast control */
    int beta = 0;       /*< Simple brightness control */

    /// Initialize values
    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
    cout << "* Enter the beta value [0-100]: ";    cin >> beta;
    //! [basic-linear-transform-parameters]

    /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
    /// Instead of these 'for' loops we could have used simply:
    /// image.convertTo(new_image, -1, alpha, beta);
    /// but we wanted to show you how to access the pixels :)
    //! [basic-linear-transform-operation]
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                new_image.at<Vec3b>(y,x)[c] =
                  saturate_cast<uchar>( alpha*image.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }
    //! [basic-linear-transform-operation]

    //! [basic-linear-transform-display]
    /// Show stuff
    imshow("Original Image", image);
    imshow("New Image", new_image);

    /// Wait until the user press a key
    waitKey();
    //! [basic-linear-transform-display]
    return 0;
}
