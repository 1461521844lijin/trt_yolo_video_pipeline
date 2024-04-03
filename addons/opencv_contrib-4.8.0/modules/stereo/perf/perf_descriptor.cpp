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
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size, MatType, MatDepth> descript_params_t;
typedef perf::TestBaseWithParam<descript_params_t> descript_params;

PERF_TEST_P( descript_params, census_sparse_descriptor,
            testing::Combine(
            testing::Values(  TYPICAL_MAT_SIZES ),
            testing::Values( CV_8U ),
            testing::Values( CV_32SC4,CV_32S )
            )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());
    Mat left(sz, matType);
    Mat out1(sz, sdepth);
    declare.in(left, WARMUP_RNG)
        .out(out1)
        .time(0.01);
    TEST_CYCLE()
    {
        censusTransform(left,9,out1,CV_SPARSE_CENSUS);
    }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P( descript_params, star_census_transform,
            testing::Combine(
            testing::Values( TYPICAL_MAT_SIZES ),
            testing::Values( CV_8U ),
            testing::Values( CV_32SC4,CV_32S )
            )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());
    Mat left(sz, matType);
    Mat out1(sz, sdepth);
    declare.in(left, WARMUP_RNG)
        .out(out1)
        .time(0.01);
    TEST_CYCLE()
    {
        starCensusTransform(left,9,out1);
    }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P( descript_params, modified_census_transform,
            testing::Combine(
            testing::Values( TYPICAL_MAT_SIZES ),
            testing::Values( CV_8U ),
            testing::Values( CV_32SC4,CV_32S )
            )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());

    Mat left(sz, matType);
    Mat out1(sz, sdepth);

    declare.in(left, WARMUP_RNG)
        .out(out1)
        .time(0.01);
    TEST_CYCLE()
    {
        modifiedCensusTransform(left,9,out1,CV_MODIFIED_CENSUS_TRANSFORM);
    }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P( descript_params, center_symetric_census,
            testing::Combine(
            testing::Values( TYPICAL_MAT_SIZES ),
            testing::Values( CV_8U ),
            testing::Values( CV_32SC4,CV_32S )
            )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());

    Mat left(sz, matType);
    Mat out1(sz, sdepth);

    declare.in(left, WARMUP_RNG)
        .out(out1)
        .time(0.01);
    TEST_CYCLE()
    {
        symetricCensusTransform(left,7,out1,CV_CS_CENSUS);
    }
    SANITY_CHECK_NOTHING();
}


}} // namespace
