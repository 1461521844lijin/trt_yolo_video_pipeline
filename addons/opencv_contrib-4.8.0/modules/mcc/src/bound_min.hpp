// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _MCC_BOUND_MIN_HPP
#define _MCC_BOUND_MIN_HPP

#include "charts.hpp"

namespace cv
{

namespace mcc
{

class CBoundMin
{

public:
    CBoundMin();
    ~CBoundMin();

    void setCharts(const std::vector<CChart> &chartIn) { chart = chartIn; }
    void getCorners(std::vector<cv::Point2f> &cornersOut) { cornersOut = corners; }
    void calculate();

private:
    std::vector<CChart> chart;
    std::vector<cv::Point2f> corners;

private:
    bool validateLine(const std::vector<cv::Point3f> &Lc, cv::Point3f ln,
                      int k, int &j)
    {

        double theta;
        cv::Point2d v0, v1;

        for (j = 0; j < k; j++)
        {
            v0.x = Lc[j].x;
            v0.y = Lc[j].y;
            v1.x = ln.x;
            v1.y = ln.y;
            theta = v0.dot(v1) / (norm(v0) * norm(v1));
            theta = acos(theta);

            if (theta < 0.5)
                return true;
        }

        return false;
    }
};

} // namespace mcc

} // namespace cv
#endif //_MCC_BOUND_MIN_HPP
