/*///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include "tldUtils.hpp"


namespace cv {
inline namespace tracking {
namespace impl {
namespace tld {

//Debug functions and variables
Rect2d etalon(14.0, 110.0, 20.0, 20.0);
void myassert(const Mat& img)
{
    int count = 0;
    for( int i = 0; i < img.rows; i++ )
    {
        for( int j = 0; j < img.cols; j++ )
        {
            if( img.at<uchar>(i, j) == 0 )
                count++;
        }
    }
    dprintf(("black: %d out of %d (%f)\n", count, img.rows * img.cols, 1.0 * count / img.rows / img.cols));
}
void printPatch(const Mat_<uchar>& standardPatch)
{
    for( int i = 0; i < standardPatch.rows; i++ )
    {
        for( int j = 0; j < standardPatch.cols; j++ )
            dprintf(("%5.2f, ", (double)standardPatch(i, j)));
        dprintf(("\n"));
    }
}
std::string type2str(const Mat& mat)
{
  int type = mat.type();
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = (uchar)(1 + (type >> CV_CN_SHIFT));

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

//Scale & Blur image using scale Indx
double scaleAndBlur(const Mat& originalImg, int scale, Mat& scaledImg, Mat& blurredImg, Size GaussBlurKernelSize, double scaleStep)
{
    double dScale = 1.0;
    for( int i = 0; i < scale; i++, dScale *= scaleStep );
    Size2d size = originalImg.size();
    size.height /= dScale; size.width /= dScale;
    resize(originalImg, scaledImg, size, 0, 0, INTER_LINEAR_EXACT);
    GaussianBlur(scaledImg, blurredImg, GaussBlurKernelSize, 0.0);
    return dScale;
}

//Find N-closest BB to the target
void getClosestN(std::vector<Rect2d>& scanGrid, Rect2d bBox, int n, std::vector<Rect2d>& res)
{
    if( n >= (int)scanGrid.size() )
    {
        res.assign(scanGrid.begin(), scanGrid.end());
        return;
    }
    std::vector<double> overlaps;
    overlaps.assign(n, 0.0);
    res.assign(scanGrid.begin(), scanGrid.begin() + n);
    for( int i = 0; i < n; i++ )
        overlaps[i] = overlap(res[i], bBox);
    double otmp;
    Rect2d rtmp;
    for (int i = 1; i < n; i++)
    {
        int j = i;
        while (j > 0 && overlaps[j - 1] > overlaps[j]) {
            otmp = overlaps[j]; overlaps[j] = overlaps[j - 1]; overlaps[j - 1] = otmp;
            rtmp = res[j]; res[j] = res[j - 1]; res[j - 1] = rtmp;
            j--;
        }
    }

    for( int i = n; i < (int)scanGrid.size(); i++ )
    {
        double o = 0.0;
        if( (o = overlap(scanGrid[i], bBox)) <= overlaps[0] )
            continue;
        int j = 0;
        while( j < n && overlaps[j] < o )
            j++;
        j--;
        for( int k = 0; k < j; overlaps[k] = overlaps[k + 1], res[k] = res[k + 1], k++ );
        overlaps[j] = o; res[j] = scanGrid[i];
    }
}

//Calculate patch variance
double variance(const Mat& img)
{
    double p = 0, p2 = 0;
    p = sum(img)(0);
    p2 = norm(img, NORM_L2SQR);
    p /= (img.cols * img.rows);
    p2 /= (img.cols * img.rows);

    return p2 - p * p;
}

//Overlap between two BB
double overlap(const Rect2d& r1, const Rect2d& r2)
{
    double a1 = r1.area(), a2 = r2.area(), a0 = (r1&r2).area();
    return a0 / (a1 + a2 - a0);
}

void resample(const Mat& img, const RotatedRect& r2, Mat_<uchar>& samples)
{
    Mat_<float> M(2, 3), R(2, 2), Si(2, 2), s(2, 1), o(2, 1);
    R(0, 0) = (float)cos(r2.angle * CV_PI / 180); R(0, 1) = (float)(-sin(r2.angle * CV_PI / 180));
    R(1, 0) = (float)sin(r2.angle * CV_PI / 180); R(1, 1) = (float)cos(r2.angle * CV_PI / 180);
    Si(0, 0) = (float)(samples.cols / r2.size.width); Si(0, 1) = 0.0f;
    Si(1, 0) = 0.0f; Si(1, 1) = (float)(samples.rows / r2.size.height);
    s(0, 0) = (float)samples.cols; s(1, 0) = (float)samples.rows;
    o(0, 0) = r2.center.x; o(1, 0) = r2.center.y;
    Mat_<float> A(2, 2), b(2, 1);
    A = Si * R;
    b = s / 2.0 - Si * R * o;
    A.copyTo(M.colRange(Range(0, 2)));
    b.copyTo(M.colRange(Range(2, 3)));
    warpAffine(img, samples, M, samples.size());
}

void resample(const Mat& img, const Rect2d& r2, Mat_<uchar>& samples)
{
    Mat_<float> M(2, 3);
    M(0, 0) = (float)(samples.cols / r2.width); M(0, 1) = 0.0f; M(0, 2) = (float)(-r2.x * samples.cols / r2.width);
    M(1, 0) = 0.0f; M(1, 1) = (float)(samples.rows / r2.height); M(1, 2) = (float)(-r2.y * samples.rows / r2.height);
    warpAffine(img, samples, M, samples.size());
}


}}}}  // namespace
