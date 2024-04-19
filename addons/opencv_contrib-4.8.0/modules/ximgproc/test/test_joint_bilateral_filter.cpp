// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

static std::string getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

static void checkSimilarity(InputArray src, InputArray ref)
{
    double normInf = cvtest::norm(src, ref, NORM_INF);
    double normL2 = cvtest::norm(src, ref, NORM_L2) / (src.total()*src.channels());

    EXPECT_LE(normInf, 1.0);
    EXPECT_LE(normL2, 1.0 / 16);
}

static Mat convertTypeAndSize(Mat src, int dstType, Size dstSize)
{
    Mat dst;
    int srcCnNum = src.channels();
    int dstCnNum = CV_MAT_CN(dstType);

    if (srcCnNum == dstCnNum)
    {
        src.copyTo(dst);
    }
    else if (srcCnNum == 3 && dstCnNum == 1)
    {
        cvtColor(src, dst, COLOR_BGR2GRAY);
    }
    else if (srcCnNum == 1 && dstCnNum == 3)
    {
        cvtColor(src, dst, COLOR_GRAY2BGR);
    }
    else
    {
        CV_Error(Error::BadNumChannels, "Bad num channels in src");
    }

    dst.convertTo(dst, dstType);
    resize(dst, dst, dstSize, 0, 0, INTER_LINEAR_EXACT);

    return dst;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void jointBilateralFilterNaive(InputArray joint, InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType = BORDER_DEFAULT);

typedef Vec<float, 1> Vec1f;
typedef Vec<uchar, 1> Vec1b;

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

template<typename T, int cn>
float normL1Sqr(const Vec<T, cn>& a, const Vec<T, cn>& b)
{
    float res = 0.0f;
    for (int i = 0; i < cn; i++)
        res += std::abs((float)a[i] - (float)b[i]);
    return res*res;
}

template<typename JointVec, typename SrcVec>
void jointBilateralFilterNaive_(InputArray joint_, InputArray src_, OutputArray dst_, int d, double sigmaColor, double sigmaSpace, int borderType)
{
    CV_Assert(joint_.size() == src_.size());
    CV_Assert(joint_.type() == traits::Type<JointVec>::value && src_.type() == traits::Type<SrcVec>::value);
    typedef Vec<float, SrcVec::channels> SrcVecf;

    if (sigmaColor <= 0)
        sigmaColor = 1;
    if (sigmaSpace <= 0)
        sigmaSpace = 1;

    int radius;
    if (d <= 0)
        radius = cvRound(sigmaSpace*1.5);
    else
        radius = d / 2;
    radius = std::max(radius, 1);
    d = 2 * radius + 1;

    dst_.create(src_.size(), src_.type());
    Mat_<SrcVec> dst = dst_.getMat();
    Mat_<JointVec> jointExt;
    Mat_<SrcVec> srcExt;
    cv::copyMakeBorder(src_, srcExt, radius, radius, radius, radius, borderType);  // TODO cvtest
    cv::copyMakeBorder(joint_, jointExt, radius, radius, radius, radius, borderType);  // TODO cvtest

    float colorGaussCoef = (float)(-0.5 / (sigmaColor*sigmaColor));
    float spaceGaussCoef = (float)(-0.5 / (sigmaSpace*sigmaSpace));

    for (int i = radius; i < srcExt.rows - radius; i++)
    {
        for (int j = radius; j < srcExt.cols - radius; j++)
        {
            JointVec joint0 = jointExt(i, j);
            SrcVecf sum = SrcVecf::all(0.0f);
            float sumWeights = 0.0f;

            for (int k = -radius; k <= radius; k++)
            {
                for (int l = -radius; l <= radius; l++)
                {
                    float spatDistSqr = (float)(k*k + l*l);
                    if (spatDistSqr > SQR(radius)) continue;

                    float colorDistSqr = normL1Sqr(joint0, jointExt(i + k, j + l));

                    float weight = std::exp(spatDistSqr*spaceGaussCoef + colorDistSqr*colorGaussCoef);

                    sum += weight*SrcVecf(srcExt(i + k, j + l));
                    sumWeights += weight;
                }
            }

            dst(i - radius, j - radius) = sum / sumWeights;
        }
    }
}

void jointBilateralFilterNaive(InputArray joint, InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType)
{
    CV_Assert(src.size() == joint.size() && src.depth() == joint.depth());
    CV_Assert(src.type() == CV_32FC1 || src.type() == CV_32FC3 || src.type() == CV_8UC1 || src.type() == CV_8UC3);
    CV_Assert(joint.type() == CV_32FC1 || joint.type() == CV_32FC3 || joint.type() == CV_8UC1 || joint.type() == CV_8UC3);

    int jointType = joint.type();
    int srcType = src.type();

    #define JBF_naive(VecJoint, VecSrc) jointBilateralFilterNaive_<VecJoint, VecSrc>(joint, src, dst, d, sigmaColor, sigmaSpace, borderType);
    if (jointType == CV_8UC1)
    {
        if (srcType == CV_8UC1)  JBF_naive(Vec1b, Vec1b);
        if (srcType == CV_8UC3)  JBF_naive(Vec1b, Vec3b);
    }
    if (jointType == CV_8UC3)
    {
        if (srcType == CV_8UC1)  JBF_naive(Vec3b, Vec1b);
        if (srcType == CV_8UC3)  JBF_naive(Vec3b, Vec3b);
    }
    if (jointType == CV_32FC1)
    {
        if (srcType == CV_32FC1) JBF_naive(Vec1f, Vec1f);
        if (srcType == CV_32FC3) JBF_naive(Vec1f, Vec3f);
    }
    if (jointType == CV_32FC3)
    {
        if (srcType == CV_32FC1) JBF_naive(Vec3f, Vec1f);
        if (srcType == CV_32FC3) JBF_naive(Vec3f, Vec3f);
    }
    #undef JBF_naive
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

typedef tuple<string, string> JBFTestParam;
typedef TestWithParam<JBFTestParam> JointBilateralFilterTest_NaiveRef;

TEST_P(JointBilateralFilterTest_NaiveRef, Accuracy)
{
    JBFTestParam param = GetParam();
    double sigmaS       = 8.0;
    string jointPath    = get<0>(param);
    string srcPath      = get<1>(param);
    int depth           = CV_8U;
    int jCn             = 3;
    int srcCn           = 1;
    int jointType       = CV_MAKE_TYPE(depth, jCn);
    int srcType         = CV_MAKE_TYPE(depth, srcCn);

    Mat joint = imread(getOpenCVExtraDir() + jointPath);
    Mat src = imread(getOpenCVExtraDir() + srcPath);
    ASSERT_TRUE(!joint.empty() && !src.empty());

    joint = convertTypeAndSize(joint, jointType, joint.size());
    src = convertTypeAndSize(src, srcType, joint.size());

    RNG rnd(cvRound(10*sigmaS) + jointType + srcType + jointPath.length() + srcPath.length() + joint.rows + joint.cols);
    double sigmaC = rnd.uniform(0, 255);

    Mat resNaive;
    jointBilateralFilterNaive(joint, src, resNaive, 0, sigmaC, sigmaS);

    Mat res;
    jointBilateralFilter(joint, src, res, 0, sigmaC, sigmaS);

    checkSimilarity(res, resNaive);
}

INSTANTIATE_TEST_CASE_P(Set2, JointBilateralFilterTest_NaiveRef,
    Combine(
    Values("/cv/shared/airplane.png", "/cv/shared/fruits.png"),
    Values("/cv/shared/airplane.png", "/cv/shared/fruits.png"))
);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

typedef tuple<string, int> BFTestParam;
typedef TestWithParam<BFTestParam> JointBilateralFilterTest_BilateralRef;

TEST_P(JointBilateralFilterTest_BilateralRef, Accuracy)
{
    BFTestParam param   = GetParam();
    double sigmaS       = 4.0;
    string srcPath      = get<0>(param);
    int srcType         = get<1>(param);

    Mat src = imread(getOpenCVExtraDir() + srcPath);
    ASSERT_TRUE(!src.empty());
    src = convertTypeAndSize(src, srcType, src.size());

    RNG rnd(cvRound(10*sigmaS) + srcPath.length() + srcType + src.rows);
    double sigmaC = rnd.uniform(0.0, 255.0);

    Mat resRef;
    bilateralFilter(src, resRef, 0, sigmaC, sigmaS);

    Mat res, joint = src.clone();
    jointBilateralFilter(joint, src, res, 0, sigmaC, sigmaS);

    checkSimilarity(res, resRef);
}

INSTANTIATE_TEST_CASE_P(Set1, JointBilateralFilterTest_BilateralRef,
    Combine(
    Values("/cv/shared/lena.png", "cv/shared/box_in_scene.png"),
    Values(CV_8UC3, CV_32FC1)
    )
);


}} // namespace
