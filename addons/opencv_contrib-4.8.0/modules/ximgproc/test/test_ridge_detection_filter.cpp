// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_ridgedetectionfilter, ReferenceAccuracy)
{
    String openCVExtraDir = cvtest::TS::ptr()->get_data_path();
    String srcImgPath = "cv/ximgproc/sources/04.png";
    String refPath = "cv/ximgproc/results/ridge_filter_test_ref/04.png";
    Mat src = imread(openCVExtraDir + srcImgPath);
    Mat ref = imread(openCVExtraDir + refPath, 0);
    Mat n_ref;
    ref.convertTo(n_ref, CV_8UC1);
    Ptr<RidgeDetectionFilter> rdf = RidgeDetectionFilter::create();
    Mat out;
    rdf->getRidgeFilteredImage(src, out);
    Mat out_cmp;
    out.convertTo(out_cmp, CV_8UC1);
    EXPECT_LE(cvtest::norm(out, ref, NORM_INF), 0.0f);
    EXPECT_LE(cvtest::norm(out, ref, NORM_L2 | NORM_RELATIVE), .0f);
}

}} // namespace
