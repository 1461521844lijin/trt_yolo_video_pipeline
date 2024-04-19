/*M///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "test_precomp.hpp"

#if defined(HAVE_CUDA) && defined(OPENCV_ENABLE_NONFREE)

namespace opencv_test { namespace {

/////////////////////////////////////////////////////////////////////////////////////////////////
// SURF

#ifdef HAVE_OPENCV_CUDAARITHM

namespace
{
    IMPLEMENT_PARAM_CLASS(SURF_HessianThreshold, double)
    IMPLEMENT_PARAM_CLASS(SURF_Octaves, int)
    IMPLEMENT_PARAM_CLASS(SURF_OctaveLayers, int)
    IMPLEMENT_PARAM_CLASS(SURF_Extended, bool)
    IMPLEMENT_PARAM_CLASS(SURF_Upright, bool)
}

PARAM_TEST_CASE(CUDA_SURF, SURF_HessianThreshold, SURF_Octaves, SURF_OctaveLayers, SURF_Extended, SURF_Upright)
{
    double hessianThreshold;
    int nOctaves;
    int nOctaveLayers;
    bool extended;
    bool upright;

    virtual void SetUp()
    {
        hessianThreshold = GET_PARAM(0);
        nOctaves = GET_PARAM(1);
        nOctaveLayers = GET_PARAM(2);
        extended = GET_PARAM(3);
        upright = GET_PARAM(4);
    }
};

CUDA_TEST_P(CUDA_SURF, Detector)
{
    cv::Mat image = readImage("../gpu/features2d/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::cuda::SURF_CUDA surf;
    surf.hessianThreshold = hessianThreshold;
    surf.nOctaves = nOctaves;
    surf.nOctaveLayers = nOctaveLayers;
    surf.extended = extended;
    surf.upright = upright;
    surf.keypointsRatio = 0.05f;

    std::vector<cv::KeyPoint> keypoints;
    surf(loadMat(image), cv::cuda::GpuMat(), keypoints);

    cv::Ptr<cv::Feature2D> surf_gold = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);

    std::vector<cv::KeyPoint> keypoints_gold;
    surf_gold->detect(image, keypoints_gold);

    int lengthDiff = abs((int)keypoints_gold.size()) - ((int)keypoints.size());
    EXPECT_LE(lengthDiff, 1);
    int matchedCount = getMatchedPointsCount(keypoints_gold, keypoints);
    double matchedRatio = static_cast<double>(matchedCount) / keypoints_gold.size();

    EXPECT_GT(matchedRatio, 0.95);
}

CUDA_TEST_P(CUDA_SURF, Detector_Masked)
{
    cv::Mat image = readImage("../gpu/features2d/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar::all(1));
    mask(cv::Range(0, image.rows / 2), cv::Range(0, image.cols / 2)).setTo(cv::Scalar::all(0));

    cv::cuda::SURF_CUDA surf;
    surf.hessianThreshold = hessianThreshold;
    surf.nOctaves = nOctaves;
    surf.nOctaveLayers = nOctaveLayers;
    surf.extended = extended;
    surf.upright = upright;
    surf.keypointsRatio = 0.05f;

    std::vector<cv::KeyPoint> keypoints;
    surf(loadMat(image), loadMat(mask), keypoints);

    cv::Ptr<cv::Feature2D> surf_gold = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);

    std::vector<cv::KeyPoint> keypoints_gold;
    surf_gold->detect(image, keypoints_gold, mask);

    int lengthDiff = abs((int)keypoints_gold.size()) - ((int)keypoints.size());
    EXPECT_LE(lengthDiff, 1);
    int matchedCount = getMatchedPointsCount(keypoints_gold, keypoints);
    double matchedRatio = static_cast<double>(matchedCount) / keypoints_gold.size();

    EXPECT_GT(matchedRatio, 0.95);
}

CUDA_TEST_P(CUDA_SURF, Descriptor)
{
    cv::Mat image = readImage("../gpu/features2d/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::cuda::SURF_CUDA surf;
    surf.hessianThreshold = hessianThreshold;
    surf.nOctaves = nOctaves;
    surf.nOctaveLayers = nOctaveLayers;
    surf.extended = extended;
    surf.upright = upright;
    surf.keypointsRatio = 0.05f;

    cv::Ptr<cv::Feature2D> surf_gold = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);

    std::vector<cv::KeyPoint> keypoints;
    surf_gold->detect(image, keypoints);

    cv::cuda::GpuMat descriptors;
    surf(loadMat(image), cv::cuda::GpuMat(), keypoints, descriptors, true);

    cv::Mat descriptors_gold;
    surf_gold->compute(image, keypoints, descriptors_gold);

    cv::BFMatcher matcher(surf.defaultNorm());
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_gold, cv::Mat(descriptors), matches);

    int matchedCount = getMatchedPointsCount(keypoints, keypoints, matches);
    double matchedRatio = static_cast<double>(matchedCount) / keypoints.size();

    EXPECT_GT(matchedRatio, 0.6);
}

testing::internal::ValueArray3<SURF_HessianThreshold, SURF_HessianThreshold, SURF_HessianThreshold> thresholdValues =
    testing::Values(
            SURF_HessianThreshold(100.0),
            SURF_HessianThreshold(500.0),
            SURF_HessianThreshold(1000.0));

INSTANTIATE_TEST_CASE_P(CUDA_Features2D, CUDA_SURF, testing::Combine(
    thresholdValues,
    testing::Values(SURF_Octaves(3), SURF_Octaves(4)),
    testing::Values(SURF_OctaveLayers(2), SURF_OctaveLayers(3)),
    testing::Values(SURF_Extended(false), SURF_Extended(true)),
    testing::Values(SURF_Upright(false), SURF_Upright(true))));

#endif // HAVE_OPENCV_CUDAARITHM

}} // namespace
#endif // HAVE_CUDA && OPENCV_ENABLE_NONFREE
