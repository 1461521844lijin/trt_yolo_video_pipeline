// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<std::string> beblid;

#define BEBLID_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

#ifdef OPENCV_ENABLE_NONFREE
PERF_TEST_P(beblid, extract, testing::Values(BEBLID_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);

    Ptr<SURF> detector = SURF::create();
    vector<KeyPoint> points;
    detector->detect(frame, points, mask);

    Ptr<BEBLID> descriptor = BEBLID::create(6.25f);
    cv::Mat descriptors;
    TEST_CYCLE() descriptor->compute(frame, points, descriptors);

    SANITY_CHECK_NOTHING();
}
#endif // NONFREE

}} // namespace
