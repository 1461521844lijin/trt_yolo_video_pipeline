// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_operators_tests.hpp"
#include <opencv2/gapi/cpu/core.hpp>

namespace
{
#define CORE_CPU [] () { return cv::compile_args(cv::gapi::use_only{cv::gapi::core::cpu::kernels()}); }
    const std::vector <cv::Size> in_sizes{ cv::Size(1280, 720), cv::Size(128, 128) };
}  // anonymous namespace

namespace opencv_test
{

// FIXME: CPU test runs are disabled since Fluid is an exclusive plugin now!
INSTANTIATE_TEST_CASE_P(MathOperatorTestCPU, MathOperatorMatMatTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values( ADD, SUB, DIV,
                                    GT, LT, GE, LE, EQ, NE)));

INSTANTIATE_TEST_CASE_P(MathOperatorTestCPU, MathOperatorMatScalarTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values( ADD,  SUB,  MUL,  DIV,
                                        ADDR, SUBR, MULR, DIVR,
                                        GT,  LT,  GE,  LE,  EQ,  NE,
                                        GTR, LTR, GER, LER, EQR, NER)));

INSTANTIATE_TEST_CASE_P(BitwiseOperatorTestCPU, MathOperatorMatMatTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values( AND, OR, XOR )));

INSTANTIATE_TEST_CASE_P(BitwiseOperatorTestCPU, MathOperatorMatScalarTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_CPU),
                                Values(AbsExact().to_compare_obj()),
                                Values( AND,  OR,  XOR,
                                        ANDR, ORR, XORR )));

INSTANTIATE_TEST_CASE_P(BitwiseNotOperatorTestCPU, NotOperatorTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_CPU)));
}
