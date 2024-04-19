// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_StructuredEdgeDetection, regression)
{
    cv::String subfolder = "cv/ximgproc/";
    cv::String dir = cvtest::TS::ptr()->get_data_path() + subfolder;
    int nTests = 12;
    float threshold = 0.01f;
    
    cv::String modelName = dir + "model.yml.gz";
    cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar =
        cv::ximgproc::createStructuredEdgeDetection(modelName);

    for (int i = 0; i < nTests; ++i)
    {
        cv::String srcName = dir + cv::format( "sources/%02d.png", i + 1);
        cv::Mat src = cv::imread( srcName, 1 );
        ASSERT_TRUE(!src.empty());

        cv::String previousResultName = dir + cv::format( "results/%02d.png", i + 1 );
        cv::Mat previousResult = cv::imread( previousResultName, 0 );
        previousResult.convertTo( previousResult, CV_32F, 1/255.0 );

        src.convertTo( src, CV_32F, 1/255.0 );

        cv::Mat currentResult( src.size(), src.type() );
        pDollar->detectEdges( src, currentResult );

        cv::Mat sqrError = ( currentResult - previousResult )
            .mul( currentResult - previousResult );
        cv::Scalar mse = cv::sum(sqrError) / cv::Scalar::all( double( sqrError.total() ) );

        EXPECT_LE( mse[0], threshold );
    }
}

}} // namespace
