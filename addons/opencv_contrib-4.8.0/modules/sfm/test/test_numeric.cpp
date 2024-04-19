/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "test_precomp.hpp"

namespace opencv_test { namespace {

template<typename T>
static void
test_meanAndVarianceAlongRows( void )
{
    int n = 4;
    Mat_<T> points(2,n);
    points << 0, 0, 1, 1,
              0, 2, 1, 3;

    Mat_<T> mean, variance;
    meanAndVarianceAlongRows(points, mean, variance);

    EXPECT_NEAR(0.5, mean(0), 1e-8);
    EXPECT_NEAR(1.5, mean(1), 1e-8);
    EXPECT_NEAR(0.25, variance(0), 1e-8);
    EXPECT_NEAR(1.25, variance(1), 1e-8);
}

TEST(Sfm_numeric, meanAndVarianceAlongRows)
{
  test_meanAndVarianceAlongRows<float>();
  test_meanAndVarianceAlongRows<double>();
}


TEST(Sfm_numeric, skewMat)
{
  // Testing with floats
  Vec3f a;
  a << 1,2,3;

  Matx33f ax = skew(a);

  EXPECT_FLOAT_EQ( 0, trace(ax) );
  EXPECT_FLOAT_EQ( ax(0,1), -ax(1,0) );
  EXPECT_FLOAT_EQ( ax(0,2), -ax(2,0) );
  EXPECT_FLOAT_EQ( ax(1,2), -ax(2,1) );

  // Testing with doubles
  Vec3d b;
  b << 1,2,3;

  Matx33d bx = skew(b);

  EXPECT_DOUBLE_EQ( 0, trace(bx) );
  EXPECT_DOUBLE_EQ( bx(0,1), -bx(1,0) );
  EXPECT_DOUBLE_EQ( bx(0,2), -bx(2,0) );
  EXPECT_DOUBLE_EQ( bx(1,2), -bx(2,1) );
}


}} // namespace
