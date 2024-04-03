/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

namespace opencv_test { namespace {

static string getDataDir() { return TS::ptr()->get_data_path(); }

static string getRubberWhaleFrame1() { return getDataDir() + "optflow/RubberWhale1.png"; }

static string getRubberWhaleFrame2() { return getDataDir() + "optflow/RubberWhale2.png"; }

static string getRubberWhaleGroundTruth() { return getDataDir() + "optflow/RubberWhale.flo"; }

static bool isFlowCorrect(float u) { return !cvIsNaN(u) && (fabs(u) < 1e9); }

static bool isFlowCorrect(double u) { return !cvIsNaN(u) && (fabs(u) < 1e9); }

static float calcRMSE(Mat flow1, Mat flow2)
{
    float sum = 0;
    int counter = 0;
    const int rows = flow1.rows;
    const int cols = flow1.cols;

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            Vec2f flow1_at_point = flow1.at<Vec2f>(y, x);
            Vec2f flow2_at_point = flow2.at<Vec2f>(y, x);

            float u1 = flow1_at_point[0];
            float v1 = flow1_at_point[1];
            float u2 = flow2_at_point[0];
            float v2 = flow2_at_point[1];

            if (isFlowCorrect(u1) && isFlowCorrect(u2) && isFlowCorrect(v1) && isFlowCorrect(v2))
            {
                sum += (u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2);
                counter++;
            }
        }
    }
    return (float)sqrt(sum / (1e-9 + counter));
}
static float calcRMSE(vector<Point2f> prevPts, vector<Point2f> currPts, Mat flow)
{
    vector<float> ee;
    for (unsigned int n = 0; n < prevPts.size(); n++)
    {
        Point2f gtFlow = flow.at<Point2f>(prevPts[n]);
        if (isFlowCorrect(gtFlow.x) && isFlowCorrect(gtFlow.y))
        {
            Point2f diffFlow = (currPts[n] - prevPts[n]) - gtFlow;
            ee.push_back(sqrt(diffFlow.x * diffFlow.x + diffFlow.y * diffFlow.y));
        }
    }
    return static_cast<float>(mean(ee).val[0]);
}
static float calcAvgEPE(vector< pair<Point2i, Point2i> > corr, Mat flow)
{
    double sum = 0;
    int counter = 0;

    for (size_t i = 0; i < corr.size(); ++i)
    {
        Vec2f flow1_at_point = Point2f(corr[i].second - corr[i].first);
        Vec2f flow2_at_point = flow.at<Vec2f>(corr[i].first.y, corr[i].first.x);

        double u1 = (double)flow1_at_point[0];
        double v1 = (double)flow1_at_point[1];
        double u2 = (double)flow2_at_point[0];
        double v2 = (double)flow2_at_point[1];

        if (isFlowCorrect(u1) && isFlowCorrect(u2) && isFlowCorrect(v1) && isFlowCorrect(v2))
        {
            sum += sqrt((u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2));
            counter++;
        }
    }

    return (float)(sum / counter);
}

bool readRubberWhale(Mat &dst_frame_1, Mat &dst_frame_2, Mat &dst_GT)
{
    string frame1_path = getRubberWhaleFrame1();
    string frame2_path = getRubberWhaleFrame2();
    string gt_flow_path = getRubberWhaleGroundTruth();
    // removing space may be an issue on windows machines
    frame1_path.erase(std::remove_if(frame1_path.begin(), frame1_path.end(), isspace), frame1_path.end());
    frame2_path.erase(std::remove_if(frame2_path.begin(), frame2_path.end(), isspace), frame2_path.end());
    gt_flow_path.erase(std::remove_if(gt_flow_path.begin(), gt_flow_path.end(), isspace), gt_flow_path.end());

    dst_frame_1 = imread(frame1_path);
    dst_frame_2 = imread(frame2_path);
    dst_GT = readOpticalFlow(gt_flow_path);

    if (dst_frame_1.empty() || dst_frame_2.empty() || dst_GT.empty())
        return false;
    else
        return true;
}


TEST(DenseOpticalFlow_SimpleFlow, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));
    float target_RMSE = 0.37f;

    Mat flow;
    Ptr<DenseOpticalFlow> algo;
    algo = createOptFlow_SimpleFlow();
    algo->calc(frame1, frame2, flow);
    ASSERT_EQ(GT.rows, flow.rows);
    ASSERT_EQ(GT.cols, flow.cols);
    EXPECT_LE(calcRMSE(GT, flow), target_RMSE);
}

TEST(DenseOpticalFlow_DeepFlow, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));
    float target_RMSE = 0.35f;
    cvtColor(frame1, frame1, COLOR_BGR2GRAY);
    cvtColor(frame2, frame2, COLOR_BGR2GRAY);

    Mat flow;
    Ptr<DenseOpticalFlow> algo;
    algo = createOptFlow_DeepFlow();
    algo->calc(frame1, frame2, flow);
    ASSERT_EQ(GT.rows, flow.rows);
    ASSERT_EQ(GT.cols, flow.cols);
    EXPECT_LE(calcRMSE(GT, flow), target_RMSE);
}

TEST(SparseOpticalFlow, ReferenceAccuracy)
{
    // with the following test each invoker class should be tested once
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));
    vector<Point2f> prevPts, currPts;
    for (int r = 0; r < frame1.rows; r+=10)
    {
        for (int c = 0; c < frame1.cols; c+=10)
        {
            prevPts.push_back(Point2f(static_cast<float>(c), static_cast<float>(r)));
        }
    }
    vector<uchar> status(prevPts.size());
    vector<float> err(prevPts.size());
    Ptr<SparseRLOFOpticalFlow> algo = SparseRLOFOpticalFlow::create();
    algo->setForwardBackward(0.0f);
    Ptr<RLOFOpticalFlowParameter> param = Ptr<RLOFOpticalFlowParameter>(new RLOFOpticalFlowParameter);
    param->supportRegionType = SR_CROSS;
    param->useIlluminationModel = true;
    param->solverType = ST_BILINEAR;
    param->setUseMEstimator(true);
    algo->setRLOFOpticalFlowParameter(param);
    algo->calc(frame1, frame2, prevPts, currPts, status, err);
    EXPECT_LE(calcRMSE(prevPts, currPts, GT), 0.3f);

    param->solverType = ST_STANDART;
    algo->setRLOFOpticalFlowParameter(param);
    algo->calc(frame1, frame2, prevPts, currPts, status, err);
    EXPECT_LE(calcRMSE(prevPts, currPts, GT), 0.34f);

    param->useIlluminationModel = false;
    param->solverType = ST_BILINEAR;
    algo->setRLOFOpticalFlowParameter(param);
    algo->calc(frame1, frame2, prevPts, currPts, status, err);
    EXPECT_LE(calcRMSE(prevPts, currPts, GT), 0.27f);

    param->solverType = ST_STANDART;
    algo->setRLOFOpticalFlowParameter(param);
    algo->calc(frame1, frame2, prevPts, currPts, status, err);
    EXPECT_LE(calcRMSE(prevPts, currPts, GT), 0.27f);

    param->setUseMEstimator(false);
    param->useIlluminationModel = true;

    param->solverType = ST_BILINEAR;
    algo->setRLOFOpticalFlowParameter(param);
    algo->calc(frame1, frame2, prevPts, currPts, status, err);
    EXPECT_LE(calcRMSE(prevPts, currPts, GT), 0.28f);

    param->solverType = ST_STANDART;
    algo->setRLOFOpticalFlowParameter(param);
    algo->calc(frame1, frame2, prevPts, currPts, status, err);
    EXPECT_LE(calcRMSE(prevPts, currPts, GT), 0.28f);

    param->useIlluminationModel = false;

    param->solverType = ST_BILINEAR;
    algo->setRLOFOpticalFlowParameter(param);
    algo->calc(frame1, frame2, prevPts, currPts, status, err);
    EXPECT_LE(calcRMSE(prevPts, currPts, GT), 0.80f);

    param->solverType = ST_STANDART;
    algo->setRLOFOpticalFlowParameter(param);
    algo->calc(frame1, frame2, prevPts, currPts, status, err);
    EXPECT_LE(calcRMSE(prevPts, currPts, GT), 0.28f);
}

TEST(DenseOpticalFlow_RLOF, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));
    Mat flow;
    Ptr<DenseRLOFOpticalFlow> algo = DenseRLOFOpticalFlow::create();
    Ptr<RLOFOpticalFlowParameter> param = Ptr<RLOFOpticalFlowParameter>(new RLOFOpticalFlowParameter);
    param->setUseMEstimator(true);
    param->supportRegionType = SR_CROSS;
    param->solverType = ST_BILINEAR;
    algo->setRLOFOpticalFlowParameter(param);
    algo->setForwardBackward(1.0f);
    algo->setGridStep(cv::Size(4, 4));
    algo->setInterpolation(INTERP_EPIC);
    algo->calc(frame1, frame2, flow);

    ASSERT_EQ(GT.rows, flow.rows);
    ASSERT_EQ(GT.cols, flow.cols);
    EXPECT_LE(calcRMSE(GT, flow), 0.46f);

    algo->setInterpolation(INTERP_GEO);
    algo->calc(frame1, frame2, flow);

    ASSERT_EQ(GT.rows, flow.rows);
    ASSERT_EQ(GT.cols, flow.cols);
    EXPECT_LE(calcRMSE(GT, flow), 0.55f);

}

TEST(DenseOpticalFlow_SparseToDenseFlow, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));
    float target_RMSE = 0.52f;

    Mat flow;
    Ptr<DenseOpticalFlow> algo;
    algo = createOptFlow_SparseToDense();
    algo->calc(frame1, frame2, flow);
    ASSERT_EQ(GT.rows, flow.rows);
    ASSERT_EQ(GT.cols, flow.cols);
    EXPECT_LE(calcRMSE(GT, flow), target_RMSE);
}

TEST(DenseOpticalFlow_PCAFlow, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));
    const float target_RMSE = 0.55f;

    Mat flow;
    Ptr<DenseOpticalFlow> algo = createOptFlow_PCAFlow();
    algo->calc(frame1, frame2, flow);
    ASSERT_EQ(GT.rows, flow.rows);
    ASSERT_EQ(GT.cols, flow.cols);
    EXPECT_LE(calcRMSE(GT, flow), target_RMSE);
}

TEST(DenseOpticalFlow_GlobalPatchColliderDCT, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));

    const Size sz = frame1.size() / 2;
    frame1 = frame1(Rect(0, 0, sz.width, sz.height));
    frame2 = frame2(Rect(0, 0, sz.width, sz.height));
    GT = GT(Rect(0, 0, sz.width, sz.height));

    vector<Mat> img1, img2, gt;
    vector< pair<Point2i, Point2i> > corr;
    img1.push_back(frame1);
    img2.push_back(frame2);
    gt.push_back(GT);

    Ptr< GPCForest<5> > forest = GPCForest<5>::create();
    forest->train(img1, img2, gt, GPCTrainingParams(8, 3, GPC_DESCRIPTOR_DCT, false));
    forest->findCorrespondences(frame1, frame2, corr);

    ASSERT_LE(7500U, corr.size());
    ASSERT_LE(calcAvgEPE(corr, GT), 0.5f);
}

TEST(DenseOpticalFlow_GlobalPatchColliderWHT, ReferenceAccuracy)
{
    Mat frame1, frame2, GT;
    ASSERT_TRUE(readRubberWhale(frame1, frame2, GT));

    const Size sz = frame1.size() / 2;
    frame1 = frame1(Rect(0, 0, sz.width, sz.height));
    frame2 = frame2(Rect(0, 0, sz.width, sz.height));
    GT = GT(Rect(0, 0, sz.width, sz.height));

    vector<Mat> img1, img2, gt;
    vector< pair<Point2i, Point2i> > corr;
    img1.push_back(frame1);
    img2.push_back(frame2);
    gt.push_back(GT);

    Ptr< GPCForest<5> > forest = GPCForest<5>::create();
    forest->train(img1, img2, gt, GPCTrainingParams(8, 3, GPC_DESCRIPTOR_WHT, false));
    forest->findCorrespondences(frame1, frame2, corr);

    ASSERT_LE(7000U, corr.size());
    ASSERT_LE(calcAvgEPE(corr, GT), 0.5f);
}


}} // namespace
