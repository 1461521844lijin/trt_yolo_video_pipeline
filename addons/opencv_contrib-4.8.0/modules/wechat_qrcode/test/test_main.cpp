// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#include "test_precomp.hpp"

#if defined(HAVE_HPX)
#include <hpx/hpx_main.hpp>
#endif

static
void initTests()
{
#ifdef HAVE_OPENCV_DNN
    const char* extraTestDataPath =
#ifdef WINRT
        NULL;
#else
        getenv("OPENCV_DNN_TEST_DATA_PATH");
#endif
    if (extraTestDataPath)
        cvtest::addDataSearchPath(extraTestDataPath);
#endif  // HAVE_OPENCV_DNN
}

CV_TEST_MAIN("cv", initTests())
