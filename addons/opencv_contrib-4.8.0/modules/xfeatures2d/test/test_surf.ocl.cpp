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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

#if defined(HAVE_OPENCL) && defined(OPENCV_ENABLE_NONFREE)

namespace opencv_test { namespace {

static bool keyPointsEquals(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
{
    const double maxPtDif = 0.1;
    const double maxSizeDif = 0.1;
    const double maxAngleDif = 0.1;
    const double maxResponseDif = 0.01;

    double dist = cv::norm(p1.pt - p2.pt);

    if (dist < maxPtDif &&
        fabs(p1.size - p2.size) < maxSizeDif &&
        abs(p1.angle - p2.angle) < maxAngleDif &&
        abs(p1.response - p2.response) < maxResponseDif &&
        p1.octave == p2.octave &&
        p1.class_id == p2.class_id)
    {
        return true;
    }

    return false;
}

static int getMatchedPointsCount(std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual)
{
    std::sort(actual.begin(), actual.end(), perf::comparators::KeypointGreater());
    std::sort(gold.begin(), gold.end(), perf::comparators::KeypointGreater());

    int validCount = 0;

    if (actual.size() == gold.size())
    {
        for (size_t i = 0; i < gold.size(); ++i)
        {
            const cv::KeyPoint& p1 = gold[i];
            const cv::KeyPoint& p2 = actual[i];

            if (keyPointsEquals(p1, p2))
                ++validCount;
        }
    }
    else
    {
        std::vector<cv::KeyPoint>& shorter = gold;
        std::vector<cv::KeyPoint>& longer = actual;
        if (actual.size() < gold.size())
        {
            shorter = actual;
            longer = gold;
        }
        for (size_t i = 0; i < shorter.size(); ++i)
        {
            const cv::KeyPoint& p1 = shorter[i];
            const cv::KeyPoint& p2 = longer[i];
            const cv::KeyPoint& p3 = longer[i+1];

            if (keyPointsEquals(p1, p2) || keyPointsEquals(p1, p3))
                ++validCount;
        }
    }

    return validCount;
}

static int getMatchedPointsCount(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches)
{
    int validCount = 0;

    for (size_t i = 0; i < matches.size(); ++i)
    {
        const cv::DMatch& m = matches[i];

        const cv::KeyPoint& p1 = keypoints1[m.queryIdx];
        const cv::KeyPoint& p2 = keypoints2[m.trainIdx];

        if (keyPointsEquals(p1, p2))
            ++validCount;
    }

    return validCount;
}

IMPLEMENT_PARAM_CLASS(HessianThreshold, double)
IMPLEMENT_PARAM_CLASS(Octaves, int)
IMPLEMENT_PARAM_CLASS(OctaveLayers, int)
IMPLEMENT_PARAM_CLASS(Extended, bool)
IMPLEMENT_PARAM_CLASS(Upright, bool)

PARAM_TEST_CASE(SURF, HessianThreshold, Octaves, OctaveLayers, Extended, Upright)
{
    bool useOpenCL;
    double hessianThreshold;
    int nOctaves;
    int nOctaveLayers;
    bool extended;
    bool upright;

    virtual void SetUp()
    {
        useOpenCL = cv::ocl::useOpenCL();
        hessianThreshold = get<0>(GetParam());
        nOctaves = get<1>(GetParam());
        nOctaveLayers = get<2>(GetParam());
        extended = get<3>(GetParam());
        upright = get<4>(GetParam());
    }

    virtual void TearDown()
    {
        cv::ocl::setUseOpenCL(useOpenCL);
    }
};

TEST_P(SURF, Detector)
{
    cv::UMat image;
    cv::ocl::setUseOpenCL(true);
    cv::imread(string(cvtest::TS::ptr()->get_data_path()) + "shared/fruits.png", cv::IMREAD_GRAYSCALE).copyTo(image);
    ASSERT_FALSE(image.empty());

    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
    std::vector<cv::KeyPoint> keypoints;
    surf->detect(image, keypoints, cv::noArray());

    cv::ocl::setUseOpenCL(false);
    std::vector<cv::KeyPoint> keypoints_gold;
    surf->detect(image, keypoints_gold, cv::noArray());

    int lengthDiff = abs((int)keypoints_gold.size()) - ((int)keypoints.size());
    EXPECT_LE(lengthDiff, 1);
    int matchedCount = getMatchedPointsCount(keypoints_gold, keypoints);
    double matchedRatio = static_cast<double>(matchedCount) / keypoints_gold.size();

    EXPECT_GT(matchedRatio, 0.99);
}

TEST_P(SURF, Descriptor)
{
    cv::UMat image;
    cv::ocl::setUseOpenCL(true);
    cv::imread(string(cvtest::TS::ptr()->get_data_path()) + "shared/fruits.png", cv::IMREAD_GRAYSCALE).copyTo(image);
    ASSERT_FALSE(image.empty());

    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);

    std::vector<cv::KeyPoint> keypoints;
    surf->detect(image, keypoints, cv::noArray());

    cv::UMat descriptors;

    surf->detectAndCompute(image, cv::noArray(), keypoints, descriptors, true);

    cv::ocl::setUseOpenCL(false);
    cv::Mat descriptors_gold;
    surf->detectAndCompute(image, cv::noArray(), keypoints, descriptors_gold, true);

    cv::BFMatcher matcher(surf->defaultNorm());
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_gold, descriptors, matches);

    int matchedCount = getMatchedPointsCount(keypoints, keypoints, matches);
    double matchedRatio = static_cast<double>(matchedCount) / keypoints.size();

    EXPECT_GT(matchedRatio, 0.35);
}

INSTANTIATE_TEST_CASE_P(OCL_Features2D, SURF, testing::Combine(
    testing::Values(HessianThreshold(500.0), HessianThreshold(1000.0)),
    testing::Values(Octaves(3), Octaves(4)),
    testing::Values(OctaveLayers(2), OctaveLayers(3)),
    testing::Values(Extended(false), Extended(true)),
    testing::Values(Upright(false), Upright(true))));

}} // namespace

#endif // HAVE_OPENCL && OPENCV_ENABLE_NONFREE
