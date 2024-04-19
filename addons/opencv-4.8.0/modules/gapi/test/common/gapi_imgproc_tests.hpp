// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_TESTS_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_HPP

#include <iostream>

#include "gapi_tests_common.hpp"

namespace opencv_test
{

// Create new value-parameterized test fixture:
// Filter2DTest - fixture name
// initMatrixRandN - function that is used to initialize input/output data
// FIXTURE_API(CompareMats,int,int) - test-specific parameters (types)
// 3 - number of test-specific parameters
// cmpF, kernSize, borderType - test-specific parameters (names)
//
// We get:
// 1. Default parameters: int type, cv::Size sz, int dtype, getCompileArgs() function
//      - available in test body
// 2. Input/output matrices will be initialized by initMatrixRandN (in this fixture)
// 3. Specific parameters: cmpF, kernSize, borderType of corresponding types
//      - created (and initialized) automatically
//      - available in test body
// Note: all parameter _values_ (e.g. type CV_8UC3) are set via INSTANTIATE_TEST_CASE_P macro
GAPI_TEST_FIXTURE(Filter2DTest, initMatrixRandN, FIXTURE_API(CompareMats,cv::Size,int), 3,
    cmpF, filterSize, borderType)
GAPI_TEST_FIXTURE(BoxFilterTest, initMatrixRandN, FIXTURE_API(CompareMats,int,int), 3,
    cmpF, filterSize, borderType)
GAPI_TEST_FIXTURE(SepFilterTest, initMatrixRandN, FIXTURE_API(CompareMats,int), 2, cmpF, kernSize)
GAPI_TEST_FIXTURE(BlurTest, initMatrixRandN, FIXTURE_API(CompareMats,int,int), 3,
    cmpF, filterSize, borderType)
GAPI_TEST_FIXTURE(GaussianBlurTest, initMatrixRandN, FIXTURE_API(CompareMats,int), 2, cmpF, kernSize)
GAPI_TEST_FIXTURE(MedianBlurTest, initMatrixRandN, FIXTURE_API(CompareMats,int), 2, cmpF, kernSize)
GAPI_TEST_FIXTURE(ErodeTest, initMatrixRandN, FIXTURE_API(CompareMats,int,int), 3,
    cmpF, kernSize, kernType)
GAPI_TEST_FIXTURE(Erode3x3Test, initMatrixRandN, FIXTURE_API(CompareMats,int), 2,
    cmpF, numIters)
GAPI_TEST_FIXTURE(DilateTest, initMatrixRandN, FIXTURE_API(CompareMats,int,int), 3,
    cmpF, kernSize, kernType)
GAPI_TEST_FIXTURE(Dilate3x3Test, initMatrixRandN, FIXTURE_API(CompareMats,int), 2, cmpF, numIters)
GAPI_TEST_FIXTURE(MorphologyExTest, initMatrixRandN, FIXTURE_API(CompareMats,cv::MorphTypes),
                  2, cmpF, op)
GAPI_TEST_FIXTURE(SobelTest, initMatrixRandN, FIXTURE_API(CompareMats,int,int,int), 4,
    cmpF, kernSize, dx, dy)
GAPI_TEST_FIXTURE(SobelXYTest, initMatrixRandN, FIXTURE_API(CompareMats,int,int,int,int), 5,
    cmpF, kernSize, order, border_type, border_val)
GAPI_TEST_FIXTURE(LaplacianTest, initMatrixRandN,
                  FIXTURE_API(CompareMats,int,double,int), 4,
                  cmpF, kernSize, scale, borderType)
GAPI_TEST_FIXTURE(BilateralFilterTest, initMatrixRandN,
                  FIXTURE_API(CompareMats,int,double,double,int), 5,
                  cmpF, d, sigmaColor, sigmaSpace, borderType)
GAPI_TEST_FIXTURE(EqHistTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(CannyTest, initMatrixRandN, FIXTURE_API(CompareMats,double,double,int,bool), 5,
    cmpF, thrLow, thrUp, apSize, l2gr)
GAPI_TEST_FIXTURE_SPEC_PARAMS(GoodFeaturesTest,
                              FIXTURE_API(CompareVectors<cv::Point2f>,std::string,int,int,double,
                                          double,int,bool),
                              8, cmpF, fileName, type, maxCorners, qualityLevel, minDistance,
                              blockSize, useHarrisDetector)
GAPI_TEST_FIXTURE_SPEC_PARAMS(FindContoursNoOffsetTest,
                              FIXTURE_API(cv::Size,MatType2,cv::RetrievalModes,
                                          cv::ContourApproximationModes, CompareMats),
                              5, sz, type, mode, method, cmpF)
GAPI_TEST_FIXTURE_SPEC_PARAMS(FindContoursOffsetTest, <>, 0)
GAPI_TEST_FIXTURE_SPEC_PARAMS(FindContoursHNoOffsetTest,
                              FIXTURE_API(cv::Size,MatType2,cv::RetrievalModes,
                                          cv::ContourApproximationModes, CompareMats),
                              5, sz, type, mode, method, cmpF)
GAPI_TEST_FIXTURE_SPEC_PARAMS(FindContoursHOffsetTest, <>, 0)
GAPI_TEST_FIXTURE(BoundingRectMatTest, initNothing, FIXTURE_API(CompareRects,bool),
                  2, cmpF, initByVector)
GAPI_TEST_FIXTURE(BoundingRectVector32STest, initNothing, FIXTURE_API(CompareRects), 1, cmpF)
GAPI_TEST_FIXTURE(BoundingRectVector32FTest, initNothing, FIXTURE_API(CompareRects), 1, cmpF)
GAPI_TEST_FIXTURE(FitLine2DMatVectorTest, initMatByPointsVectorRandU<cv::Point_>,
                  FIXTURE_API(CompareVecs<float, 4>,cv::DistanceTypes), 2, cmpF, distType)
GAPI_TEST_FIXTURE(FitLine2DVector32STest, initNothing,
                  FIXTURE_API(CompareVecs<float, 4>,cv::DistanceTypes), 2, cmpF, distType)
GAPI_TEST_FIXTURE(FitLine2DVector32FTest, initNothing,
                  FIXTURE_API(CompareVecs<float, 4>,cv::DistanceTypes), 2, cmpF, distType)
GAPI_TEST_FIXTURE(FitLine2DVector64FTest, initNothing,
                  FIXTURE_API(CompareVecs<float, 4>,cv::DistanceTypes), 2, cmpF, distType)
GAPI_TEST_FIXTURE(FitLine3DMatVectorTest, initMatByPointsVectorRandU<cv::Point3_>,
                  FIXTURE_API(CompareVecs<float, 6>,cv::DistanceTypes), 2, cmpF, distType)
GAPI_TEST_FIXTURE(FitLine3DVector32STest, initNothing,
                  FIXTURE_API(CompareVecs<float, 6>,cv::DistanceTypes), 2, cmpF, distType)
GAPI_TEST_FIXTURE(FitLine3DVector32FTest, initNothing,
                  FIXTURE_API(CompareVecs<float, 6>,cv::DistanceTypes), 2, cmpF, distType)
GAPI_TEST_FIXTURE(FitLine3DVector64FTest, initNothing,
                  FIXTURE_API(CompareVecs<float, 6>,cv::DistanceTypes), 2, cmpF, distType)
GAPI_TEST_FIXTURE(BGR2RGBTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(RGB2GrayTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(BGR2GrayTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(RGB2YUVTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(BGR2I420Test, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(RGB2I420Test, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(I4202BGRTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(I4202RGBTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(YUV2RGBTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(YUV2GrayTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(NV12toRGBTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(NV12toBGRpTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(NV12toRGBpTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(NV12toBGRTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(NV12toGrayTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(RGB2LabTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(BGR2LUVTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(LUV2BGRTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(BGR2YUVTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(YUV2BGRTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(RGB2HSVTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(BayerGR2RGBTest, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(RGB2YUV422Test, initMatrixRandN, FIXTURE_API(CompareMats), 1, cmpF)
GAPI_TEST_FIXTURE(ResizeTest, initNothing, FIXTURE_API(CompareMats,int,cv::Size), 3,
    cmpF, interp, sz_out)
GAPI_TEST_FIXTURE(ResizePTest, initNothing, FIXTURE_API(CompareMats,int,cv::Size), 3,
    cmpF, interp, sz_out)
GAPI_TEST_FIXTURE(ResizeTestFxFy, initNothing, FIXTURE_API(CompareMats,int,double,double), 4,
    cmpF, interp, fx, fy)
} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_TESTS_HPP
