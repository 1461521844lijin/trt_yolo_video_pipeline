// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include <opencv2/calib3d.hpp>

#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/aruco_board.hpp"
#include "apriltag/apriltag_quad_thresh.hpp"
#include "aruco_utils.hpp"
#include <cmath>

namespace cv {
namespace aruco {

using namespace std;

static inline bool readWrite(DetectorParameters &params, const FileNode* readNode,
                             FileStorage* writeStorage = nullptr)
{
    CV_Assert(readNode || writeStorage);
    bool check = false;

    check |= readWriteParameter("adaptiveThreshWinSizeMin", params.adaptiveThreshWinSizeMin, readNode, writeStorage);
    check |= readWriteParameter("adaptiveThreshWinSizeMax", params.adaptiveThreshWinSizeMax, readNode, writeStorage);
    check |= readWriteParameter("adaptiveThreshWinSizeStep", params.adaptiveThreshWinSizeStep, readNode, writeStorage);
    check |= readWriteParameter("adaptiveThreshConstant", params.adaptiveThreshConstant, readNode, writeStorage);
    check |= readWriteParameter("minMarkerPerimeterRate", params.minMarkerPerimeterRate, readNode, writeStorage);
    check |= readWriteParameter("maxMarkerPerimeterRate", params.maxMarkerPerimeterRate, readNode, writeStorage);
    check |= readWriteParameter("polygonalApproxAccuracyRate", params.polygonalApproxAccuracyRate,
                                readNode, writeStorage);
    check |= readWriteParameter("minCornerDistanceRate", params.minCornerDistanceRate, readNode, writeStorage);
    check |= readWriteParameter("minDistanceToBorder", params.minDistanceToBorder, readNode, writeStorage);
    check |= readWriteParameter("minMarkerDistanceRate", params.minMarkerDistanceRate, readNode, writeStorage);
    check |= readWriteParameter("cornerRefinementMethod", params.cornerRefinementMethod, readNode, writeStorage);
    check |= readWriteParameter("cornerRefinementWinSize", params.cornerRefinementWinSize, readNode, writeStorage);
    check |= readWriteParameter("cornerRefinementMaxIterations", params.cornerRefinementMaxIterations,
                                readNode, writeStorage);
    check |= readWriteParameter("cornerRefinementMinAccuracy", params.cornerRefinementMinAccuracy,
                                readNode, writeStorage);
    check |= readWriteParameter("markerBorderBits", params.markerBorderBits, readNode, writeStorage);
    check |= readWriteParameter("perspectiveRemovePixelPerCell", params.perspectiveRemovePixelPerCell,
                                readNode, writeStorage);
    check |= readWriteParameter("perspectiveRemoveIgnoredMarginPerCell", params.perspectiveRemoveIgnoredMarginPerCell,
                                readNode, writeStorage);
    check |= readWriteParameter("maxErroneousBitsInBorderRate", params.maxErroneousBitsInBorderRate,
                                readNode, writeStorage);
    check |= readWriteParameter("minOtsuStdDev", params.minOtsuStdDev, readNode, writeStorage);
    check |= readWriteParameter("errorCorrectionRate", params.errorCorrectionRate, readNode, writeStorage);
    // new aruco 3 functionality
    check |= readWriteParameter("useAruco3Detection", params.useAruco3Detection, readNode, writeStorage);
    check |= readWriteParameter("minSideLengthCanonicalImg", params.minSideLengthCanonicalImg, readNode, writeStorage);
    check |= readWriteParameter("minMarkerLengthRatioOriginalImg", params.minMarkerLengthRatioOriginalImg,
                                readNode, writeStorage);
    return check;
}

bool DetectorParameters::readDetectorParameters(const FileNode& fn)
{
    if (fn.empty())
        return false;
    return readWrite(*this, &fn);
}

bool DetectorParameters::writeDetectorParameters(FileStorage& fs, const String& name)
{
    CV_Assert(fs.isOpened());
    if (!name.empty())
        fs << name << "{";
    bool res = readWrite(*this, nullptr, &fs);
    if (!name.empty())
        fs << "}";
    return res;
}

static inline bool readWrite(RefineParameters& refineParameters, const FileNode* readNode,
                             FileStorage* writeStorage = nullptr)
{
    CV_Assert(readNode || writeStorage);
    bool check = false;

    check |= readWriteParameter("minRepDistance", refineParameters.minRepDistance, readNode, writeStorage);
    check |= readWriteParameter("errorCorrectionRate", refineParameters.errorCorrectionRate, readNode, writeStorage);
    check |= readWriteParameter("checkAllOrders", refineParameters.checkAllOrders, readNode, writeStorage);
    return check;
}

RefineParameters::RefineParameters(float _minRepDistance, float _errorCorrectionRate, bool _checkAllOrders):
                                   minRepDistance(_minRepDistance), errorCorrectionRate(_errorCorrectionRate),
                                   checkAllOrders(_checkAllOrders){}

bool RefineParameters::readRefineParameters(const FileNode &fn)
{
    if (fn.empty())
        return false;
    return readWrite(*this, &fn);
}

bool RefineParameters::writeRefineParameters(FileStorage& fs, const String& name)
{
    CV_Assert(fs.isOpened());
    if (!name.empty())
        fs << name << "{";
    bool res = readWrite(*this, nullptr, &fs);
    if (!name.empty())
        fs << "}";
    return res;
}

/**
  * @brief Threshold input image using adaptive thresholding
  */
static void _threshold(InputArray _in, OutputArray _out, int winSize, double constant) {

    CV_Assert(winSize >= 3);
    if(winSize % 2 == 0) winSize++; // win size must be odd
    adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, winSize, constant);
}


/**
  * @brief Given a tresholded image, find the contours, calculate their polygonal approximation
  * and take those that accomplish some conditions
  */
static void _findMarkerContours(const Mat &in, vector<vector<Point2f> > &candidates,
                                vector<vector<Point> > &contoursOut, double minPerimeterRate,
                                double maxPerimeterRate, double accuracyRate,
                                double minCornerDistanceRate, int minDistanceToBorder, int minSize) {

    CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 &&
              minCornerDistanceRate >= 0 && minDistanceToBorder >= 0);

    // calculate maximum and minimum sizes in pixels
    unsigned int minPerimeterPixels =
        (unsigned int)(minPerimeterRate * max(in.cols, in.rows));
    unsigned int maxPerimeterPixels =
        (unsigned int)(maxPerimeterRate * max(in.cols, in.rows));

    // for aruco3 functionality
    if (minSize != 0) {
        minPerimeterPixels = 4*minSize;
    }

    Mat contoursImg;
    in.copyTo(contoursImg);
    vector<vector<Point> > contours;
    findContours(contoursImg, contours, RETR_LIST, CHAIN_APPROX_NONE);
    // now filter list of contours
    for(unsigned int i = 0; i < contours.size(); i++) {
        // check perimeter
        if(contours[i].size() < minPerimeterPixels || contours[i].size() > maxPerimeterPixels)
            continue;

        // check is square and is convex
        vector<Point> approxCurve;
        approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * accuracyRate, true);
        if(approxCurve.size() != 4 || !isContourConvex(approxCurve)) continue;

        // check min distance between corners
        double minDistSq =
            max(contoursImg.cols, contoursImg.rows) * max(contoursImg.cols, contoursImg.rows);
        for(int j = 0; j < 4; j++) {
            double d = (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                           (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                           (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y);
            minDistSq = min(minDistSq, d);
        }
        double minCornerDistancePixels = double(contours[i].size()) * minCornerDistanceRate;
        if(minDistSq < minCornerDistancePixels * minCornerDistancePixels) continue;

        // check if it is too near to the image border
        bool tooNearBorder = false;
        for(int j = 0; j < 4; j++) {
            if(approxCurve[j].x < minDistanceToBorder || approxCurve[j].y < minDistanceToBorder ||
               approxCurve[j].x > contoursImg.cols - 1 - minDistanceToBorder ||
               approxCurve[j].y > contoursImg.rows - 1 - minDistanceToBorder)
                tooNearBorder = true;
        }
        if(tooNearBorder) continue;

        // if it passes all the test, add to candidates vector
        vector<Point2f> currentCandidate;
        currentCandidate.resize(4);
        for(int j = 0; j < 4; j++) {
            currentCandidate[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
        }
        candidates.push_back(currentCandidate);
        contoursOut.push_back(contours[i]);
    }
}


/**
  * @brief Assure order of candidate corners is clockwise direction
  */
static void _reorderCandidatesCorners(vector<vector<Point2f> > &candidates) {

    for(unsigned int i = 0; i < candidates.size(); i++) {
        double dx1 = candidates[i][1].x - candidates[i][0].x;
        double dy1 = candidates[i][1].y - candidates[i][0].y;
        double dx2 = candidates[i][2].x - candidates[i][0].x;
        double dy2 = candidates[i][2].y - candidates[i][0].y;
        double crossProduct = (dx1 * dy2) - (dy1 * dx2);

        if(crossProduct < 0.0) { // not clockwise direction
            swap(candidates[i][1], candidates[i][3]);
        }
    }
}

/**
  * @brief to make sure that the corner's order of both candidates (default/white) is the same
  */
static vector<Point2f> alignContourOrder(Point2f corner, vector<Point2f> candidate) {
    uint8_t r=0;
    double min = norm( Vec2f( corner - candidate[0] ), NORM_L2SQR);
    for(uint8_t pos=1; pos < 4; pos++) {
        double nDiff = norm( Vec2f( corner - candidate[pos] ), NORM_L2SQR);
        if(nDiff < min){
            r = pos;
            min =nDiff;
        }
    }
    std::rotate(candidate.begin(), candidate.begin() + r, candidate.end());
    return candidate;
}

/**
  * @brief Check candidates that are too close to each other, save the potential candidates
  *        (i.e. biggest/smallest contour) and remove the rest
  */
static void _filterTooCloseCandidates(const vector<vector<Point2f> > &candidatesIn,
                                      vector<vector<vector<Point2f> > > &candidatesSetOut,
                                      const vector<vector<Point> > &contoursIn,
                                      vector<vector<vector<Point> > > &contoursSetOut,
                                      double minMarkerDistanceRate, bool detectInvertedMarker) {

    CV_Assert(minMarkerDistanceRate >= 0);
    vector<int> candGroup;
    candGroup.resize(candidatesIn.size(), -1);
    vector<vector<unsigned int> > groupedCandidates;
    for(unsigned int i = 0; i < candidatesIn.size(); i++) {
        bool isSingleContour = true;
        for(unsigned int j = i + 1; j < candidatesIn.size(); j++) {

            int minimumPerimeter = min((int)contoursIn[i].size(), (int)contoursIn[j].size() );

            // fc is the first corner considered on one of the markers, 4 combinations are possible
            for(int fc = 0; fc < 4; fc++) {
                double distSq = 0;
                for(int c = 0; c < 4; c++) {
                    // modC is the corner considering first corner is fc
                    int modC = (c + fc) % 4;
                    distSq += (candidatesIn[i][modC].x - candidatesIn[j][c].x) *
                                  (candidatesIn[i][modC].x - candidatesIn[j][c].x) +
                              (candidatesIn[i][modC].y - candidatesIn[j][c].y) *
                                  (candidatesIn[i][modC].y - candidatesIn[j][c].y);
                }
                distSq /= 4.;

                // if mean square distance is too low, remove the smaller one of the two markers
                double minMarkerDistancePixels = double(minimumPerimeter) * minMarkerDistanceRate;
                if(distSq < minMarkerDistancePixels * minMarkerDistancePixels) {
                    isSingleContour = false;
                    // i and j are not related to a group
                    if(candGroup[i]<0 && candGroup[j]<0){
                        // mark candidates with their corresponding group number
                        candGroup[i] = candGroup[j] = (int)groupedCandidates.size();

                        // create group
                        vector<unsigned int> grouped;
                        grouped.push_back(i);
                        grouped.push_back(j);
                        groupedCandidates.push_back( grouped );
                    }
                    // i is related to a group
                    else if(candGroup[i] > -1 && candGroup[j] == -1){
                        int group = candGroup[i];
                        candGroup[j] = group;

                        // add to group
                        groupedCandidates[group].push_back( j );
                    }
                    // j is related to a group
                    else if(candGroup[j] > -1 && candGroup[i] == -1){
                        int group = candGroup[j];
                        candGroup[i] = group;

                        // add to group
                        groupedCandidates[group].push_back( i );
                    }
                }
            }
        }
        if (isSingleContour && candGroup[i] < 0)
        {
            candGroup[i] = (int)groupedCandidates.size();
            vector<unsigned int> grouped;
            grouped.push_back(i);
            grouped.push_back(i); // step "save possible candidates" require minimum 2 elements
            groupedCandidates.push_back(grouped);
        }
    }

    // save possible candidates
    candidatesSetOut.clear();
    contoursSetOut.clear();

    vector<vector<Point2f> > biggerCandidates;
    vector<vector<Point> > biggerContours;
    vector<vector<Point2f> > smallerCandidates;
    vector<vector<Point> > smallerContours;

    // save possible candidates
    for(unsigned int i = 0; i < groupedCandidates.size(); i++) {
        unsigned int smallerIdx = groupedCandidates[i][0];
        unsigned int biggerIdx = smallerIdx;
        double smallerArea = contourArea(candidatesIn[smallerIdx]);
        double biggerArea = smallerArea;

        // evaluate group elements
        for(unsigned int j = 1; j < groupedCandidates[i].size(); j++) {
            unsigned int currIdx = groupedCandidates[i][j];
            double currArea = contourArea(candidatesIn[currIdx]);

            // check if current contour is bigger
            if(currArea >= biggerArea) {
                biggerIdx = currIdx;
                biggerArea = currArea;
            }

            // check if current contour is smaller
            if(currArea < smallerArea && detectInvertedMarker) {
                smallerIdx = currIdx;
                smallerArea = currArea;
            }
        }

        // add contours and candidates
        biggerCandidates.push_back(candidatesIn[biggerIdx]);
        biggerContours.push_back(contoursIn[biggerIdx]);
        if(detectInvertedMarker) {
            smallerCandidates.push_back(alignContourOrder(candidatesIn[biggerIdx][0], candidatesIn[smallerIdx]));
            smallerContours.push_back(contoursIn[smallerIdx]);
        }
    }
    // to preserve the structure :: candidateSet< defaultCandidates, whiteCandidates >
    // default candidates
    candidatesSetOut.push_back(biggerCandidates);
    contoursSetOut.push_back(biggerContours);
    // white candidates
    candidatesSetOut.push_back(smallerCandidates);
    contoursSetOut.push_back(smallerContours);
}

/**
 * @brief Initial steps on finding square candidates
 */
static void _detectInitialCandidates(const Mat &grey, vector<vector<Point2f> > &candidates,
                                     vector<vector<Point> > &contours,
                                     const DetectorParameters &params) {

    CV_Assert(params.adaptiveThreshWinSizeMin >= 3 && params.adaptiveThreshWinSizeMax >= 3);
    CV_Assert(params.adaptiveThreshWinSizeMax >= params.adaptiveThreshWinSizeMin);
    CV_Assert(params.adaptiveThreshWinSizeStep > 0);

    // number of window sizes (scales) to apply adaptive thresholding
    int nScales =  (params.adaptiveThreshWinSizeMax - params.adaptiveThreshWinSizeMin) /
                      params.adaptiveThreshWinSizeStep + 1;

    vector<vector<vector<Point2f> > > candidatesArrays((size_t) nScales);
    vector<vector<vector<Point> > > contoursArrays((size_t) nScales);

    ////for each value in the interval of thresholding window sizes
    parallel_for_(Range(0, nScales), [&](const Range& range) {
        const int begin = range.start;
        const int end = range.end;

        for (int i = begin; i < end; i++) {
            int currScale = params.adaptiveThreshWinSizeMin + i * params.adaptiveThreshWinSizeStep;
            // threshold
            Mat thresh;
            _threshold(grey, thresh, currScale, params.adaptiveThreshConstant);

            // detect rectangles
            _findMarkerContours(thresh, candidatesArrays[i], contoursArrays[i],
                                params.minMarkerPerimeterRate, params.maxMarkerPerimeterRate,
                                params.polygonalApproxAccuracyRate, params.minCornerDistanceRate,
                                params.minDistanceToBorder, params.minSideLengthCanonicalImg);
        }
    });
    // join candidates
    for(int i = 0; i < nScales; i++) {
        for(unsigned int j = 0; j < candidatesArrays[i].size(); j++) {
            candidates.push_back(candidatesArrays[i][j]);
            contours.push_back(contoursArrays[i][j]);
        }
    }
}


/**
 * @brief Detect square candidates in the input image
 */
static void _detectCandidates(InputArray _grayImage, vector<vector<vector<Point2f> > >& candidatesSetOut,
                              vector<vector<vector<Point> > >& contoursSetOut, const DetectorParameters &_params) {
    Mat grey = _grayImage.getMat();
    CV_DbgAssert(grey.total() != 0);
    CV_DbgAssert(grey.type() == CV_8UC1);

    /// 1. DETECT FIRST SET OF CANDIDATES
    vector<vector<Point2f> > candidates;
    vector<vector<Point> > contours;
    _detectInitialCandidates(grey, candidates, contours, _params);
    /// 2. SORT CORNERS
    _reorderCandidatesCorners(candidates);

    /// 3. FILTER OUT NEAR CANDIDATE PAIRS
    // save the outter/inner border (i.e. potential candidates)
    _filterTooCloseCandidates(candidates, candidatesSetOut, contours, contoursSetOut,
                              _params.minMarkerDistanceRate, _params.detectInvertedMarker);
}


/**
  * @brief Given an input image and candidate corners, extract the bits of the candidate, including
  * the border bits
  */
static Mat _extractBits(InputArray _image, const vector<Point2f>& corners, int markerSize,
                        int markerBorderBits, int cellSize, double cellMarginRate, double minStdDevOtsu) {
    CV_Assert(_image.getMat().channels() == 1);
    CV_Assert(corners.size() == 4ull);
    CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate >= 0 && cellMarginRate <= 1);
    CV_Assert(minStdDevOtsu >= 0);

    // number of bits in the marker
    int markerSizeWithBorders = markerSize + 2 * markerBorderBits;
    int cellMarginPixels = int(cellMarginRate * cellSize);

    Mat resultImg; // marker image after removing perspective
    int resultImgSize = markerSizeWithBorders * cellSize;
    Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
    resultImgCorners.ptr<Point2f>(0)[1] = Point2f((float)resultImgSize - 1, 0);
    resultImgCorners.ptr<Point2f>(0)[2] =
        Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
    resultImgCorners.ptr<Point2f>(0)[3] = Point2f(0, (float)resultImgSize - 1);

    // remove perspective
    Mat transformation = getPerspectiveTransform(corners, resultImgCorners);
    warpPerspective(_image, resultImg, transformation, Size(resultImgSize, resultImgSize),
                    INTER_NEAREST);

    // output image containing the bits
    Mat bits(markerSizeWithBorders, markerSizeWithBorders, CV_8UC1, Scalar::all(0));

    // check if standard deviation is enough to apply Otsu
    // if not enough, it probably means all bits are the same color (black or white)
    Mat mean, stddev;
    // Remove some border just to avoid border noise from perspective transformation
    Mat innerRegion = resultImg.colRange(cellSize / 2, resultImg.cols - cellSize / 2)
                          .rowRange(cellSize / 2, resultImg.rows - cellSize / 2);
    meanStdDev(innerRegion, mean, stddev);
    if(stddev.ptr< double >(0)[0] < minStdDevOtsu) {
        // all black or all white, depending on mean value
        if(mean.ptr< double >(0)[0] > 127)
            bits.setTo(1);
        else
            bits.setTo(0);
        return bits;
    }

    // now extract code, first threshold using Otsu
    threshold(resultImg, resultImg, 125, 255, THRESH_BINARY | THRESH_OTSU);

    // for each cell
    for(int y = 0; y < markerSizeWithBorders; y++) {
        for(int x = 0; x < markerSizeWithBorders; x++) {
            int Xstart = x * (cellSize) + cellMarginPixels;
            int Ystart = y * (cellSize) + cellMarginPixels;
            Mat square = resultImg(Rect(Xstart, Ystart, cellSize - 2 * cellMarginPixels,
                                        cellSize - 2 * cellMarginPixels));
            // count white pixels on each cell to assign its value
            size_t nZ = (size_t) countNonZero(square);
            if(nZ > square.total() / 2) bits.at<unsigned char>(y, x) = 1;
        }
    }

    return bits;
}



/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
static int _getBorderErrors(const Mat &bits, int markerSize, int borderSize) {

    int sizeWithBorders = markerSize + 2 * borderSize;

    CV_Assert(markerSize > 0 && bits.cols == sizeWithBorders && bits.rows == sizeWithBorders);

    int totalErrors = 0;
    for(int y = 0; y < sizeWithBorders; y++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(y)[k] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(y)[sizeWithBorders - 1 - k] != 0) totalErrors++;
        }
    }
    for(int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(k)[x] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(sizeWithBorders - 1 - k)[x] != 0) totalErrors++;
        }
    }
    return totalErrors;
}


/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
static uint8_t _identifyOneCandidate(const Dictionary& dictionary, InputArray _image,
                                     const vector<Point2f>& _corners, int& idx,
                                     const DetectorParameters& params, int& rotation,
                                     const float scale = 1.f) {
    CV_DbgAssert(_corners.size() == 4);
    CV_DbgAssert(_image.getMat().total() != 0);
    CV_DbgAssert(params.markerBorderBits > 0);
    uint8_t typ=1;
    // get bits
    // scale corners to the correct size to search on the corresponding image pyramid
    vector<Point2f> scaled_corners(4);
    for (int i = 0; i < 4; ++i) {
        scaled_corners[i].x = _corners[i].x * scale;
        scaled_corners[i].y = _corners[i].y * scale;
    }

    Mat candidateBits =
        _extractBits(_image, scaled_corners, dictionary.markerSize, params.markerBorderBits,
                     params.perspectiveRemovePixelPerCell,
                     params.perspectiveRemoveIgnoredMarginPerCell, params.minOtsuStdDev);

    // analyze border bits
    int maximumErrorsInBorder =
        int(dictionary.markerSize * dictionary.markerSize * params.maxErroneousBitsInBorderRate);
    int borderErrors =
        _getBorderErrors(candidateBits, dictionary.markerSize, params.markerBorderBits);

    // check if it is a white marker
    if(params.detectInvertedMarker){
        // to get from 255 to 1
        Mat invertedImg = ~candidateBits-254;
        int invBError = _getBorderErrors(invertedImg, dictionary.markerSize, params.markerBorderBits);
        // white marker
        if(invBError<borderErrors){
            borderErrors = invBError;
            invertedImg.copyTo(candidateBits);
            typ=2;
        }
    }
    if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong

    // take only inner bits
    Mat onlyBits =
        candidateBits.rowRange(params.markerBorderBits,
                               candidateBits.rows - params.markerBorderBits)
            .colRange(params.markerBorderBits, candidateBits.cols - params.markerBorderBits);

    // try to indentify the marker
    if(!dictionary.identify(onlyBits, idx, rotation, params.errorCorrectionRate))
        return 0;

    return typ;
}

/**
 * @brief rotate the initial corner to get to the right position
 */
static void correctCornerPosition(vector<Point2f>& _candidate, int rotate){
    std::rotate(_candidate.begin(), _candidate.begin() + 4 - rotate, _candidate.end());
}

static size_t _findOptPyrImageForCanonicalImg(
        const vector<Mat>& img_pyr,
        const int scaled_width,
        const int cur_perimeter,
        const int min_perimeter) {
    CV_Assert(scaled_width > 0);
    size_t optLevel = 0;
    float dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < img_pyr.size(); ++i) {
        const float scale = img_pyr[i].cols / static_cast<float>(scaled_width);
        const float perimeter_scaled = cur_perimeter * scale;
        // instead of std::abs() favor the larger pyramid level by checking if the distance is postive
        // will slow down the algorithm but find more corners in the end
        const float new_dist = perimeter_scaled - min_perimeter;
        if (new_dist < dist && new_dist > 0.f) {
            dist = new_dist;
            optLevel = i;
        }
    }
    return optLevel;
}

/**
 * @brief Identify square candidates according to a marker dictionary
 */

static void _identifyCandidates(InputArray grey,
                                const vector<Mat>& image_pyr,
                                vector<vector<vector<Point2f> > >& _candidatesSet,
                                vector<vector<vector<Point> > >& _contoursSet, const Dictionary &_dictionary,
                                vector<vector<Point2f> >& _accepted, vector<vector<Point> >& _contours, vector<int>& ids,
                                const DetectorParameters &params,
                                OutputArrayOfArrays _rejected = noArray()) {
    CV_DbgAssert(grey.getMat().total() != 0);
    CV_DbgAssert(grey.getMat().type() == CV_8UC1);
    int ncandidates = (int)_candidatesSet[0].size();
    vector<vector<Point2f> > accepted;
    vector<vector<Point2f> > rejected;
    vector<vector<Point> > contours;

    vector<int> idsTmp(ncandidates, -1);
    vector<int> rotated(ncandidates, 0);
    vector<uint8_t> validCandidates(ncandidates, 0);

    //// Analyze each of the candidates
    parallel_for_(Range(0, ncandidates), [&](const Range &range) {
        const int begin = range.start;
        const int end = range.end;

        vector<vector<Point2f> >& candidates = params.detectInvertedMarker ? _candidatesSet[1] : _candidatesSet[0];
        vector<vector<Point> >& contourS = params.detectInvertedMarker ? _contoursSet[1] : _contoursSet[0];

        for(int i = begin; i < end; i++) {
            int currId = -1;
            // implements equation (4)
            if (params.useAruco3Detection) {
                const int perimeterOfContour = static_cast<int>(contourS[i].size());
                const int min_perimeter = params.minSideLengthCanonicalImg * 4;
                const size_t nearestImgId = _findOptPyrImageForCanonicalImg(image_pyr, grey.cols(), perimeterOfContour, min_perimeter);
                const float scale = image_pyr[nearestImgId].cols / static_cast<float>(grey.cols());

                validCandidates[i] = _identifyOneCandidate(_dictionary, image_pyr[nearestImgId], candidates[i], currId, params, rotated[i], scale);
            }
            else {
                validCandidates[i] = _identifyOneCandidate(_dictionary, grey, candidates[i], currId, params, rotated[i]);
            }

            if(validCandidates[i] > 0)
                idsTmp[i] = currId;
        }
    });

    for(int i = 0; i < ncandidates; i++) {
        if(validCandidates[i] > 0) {
            // to choose the right set of candidates :: 0 for default, 1 for white markers
            uint8_t set = validCandidates[i]-1;

            // shift corner positions to the correct rotation
            correctCornerPosition(_candidatesSet[set][i], rotated[i]);

            if( !params.detectInvertedMarker && validCandidates[i] == 2 )
                continue;

            // add valid candidate
            accepted.push_back(_candidatesSet[set][i]);
            ids.push_back(idsTmp[i]);

            contours.push_back(_contoursSet[set][i]);

        } else {
            rejected.push_back(_candidatesSet[0][i]);
        }
    }

    // parse output
    _accepted = accepted;

    _contours= contours;

    if(_rejected.needed()) {
        _copyVector2Output(rejected, _rejected);
    }
}

/**
 * Line fitting  A * B = C :: Called from function refineCandidateLines
 * @param nContours, contour-container
 */
static Point3f _interpolate2Dline(const vector<Point2f>& nContours){
    CV_Assert(nContours.size() >= 2);
    float minX, minY, maxX, maxY;
    minX = maxX = nContours[0].x;
    minY = maxY = nContours[0].y;

    for(unsigned int i = 0; i< nContours.size(); i++){
        minX = nContours[i].x < minX ? nContours[i].x : minX;
        minY = nContours[i].y < minY ? nContours[i].y : minY;
        maxX = nContours[i].x > maxX ? nContours[i].x : maxX;
        maxY = nContours[i].y > maxY ? nContours[i].y : maxY;
    }

    Mat A = Mat::ones((int)nContours.size(), 2, CV_32F); // Coefficient Matrix (N x 2)
    Mat B((int)nContours.size(), 1, CV_32F);                // Variables   Matrix (N x 1)
    Mat C;                                            // Constant

    if(maxX - minX > maxY - minY){
        for(unsigned int i =0; i < nContours.size(); i++){
            A.at<float>(i,0)= nContours[i].x;
            B.at<float>(i,0)= nContours[i].y;
        }

        solve(A, B, C, DECOMP_NORMAL);

        return Point3f(C.at<float>(0, 0), -1., C.at<float>(1, 0));
    }
    else{
        for(unsigned int i =0; i < nContours.size(); i++){
            A.at<float>(i,0)= nContours[i].y;
            B.at<float>(i,0)= nContours[i].x;
        }

        solve(A, B, C, DECOMP_NORMAL);

        return Point3f(-1., C.at<float>(0, 0), C.at<float>(1, 0));
    }

}

/**
 * Find the Point where the lines crosses :: Called from function refineCandidateLines
 * @param nLine1
 * @param nLine2
 * @return Crossed Point
 */
static Point2f _getCrossPoint(Point3f nLine1, Point3f nLine2){
    Matx22f A(nLine1.x, nLine1.y, nLine2.x, nLine2.y);
    Vec2f B(-nLine1.z, -nLine2.z);
    return Vec2f(A.solve(B).val);
}

/**
 * Refine Corners using the contour vector :: Called from function detectMarkers
 * @param nContours, contour-container
 * @param nCorners, candidate Corners
 * @param camMatrix, cameraMatrix input 3x3 floating-point camera matrix
 * @param distCoeff, distCoeffs vector of distortion coefficient
 */
static void _refineCandidateLines(vector<Point>& nContours, vector<Point2f>& nCorners){
    vector<Point2f> contour2f(nContours.begin(), nContours.end());
    /* 5 groups :: to group the edges
     * 4 - classified by its corner
     * extra group - (temporary) if contours do not begin with a corner
     */
    vector<Point2f> cntPts[5];
    int cornerIndex[4]={-1};
    int group=4;

    for ( unsigned int i =0; i < nContours.size(); i++ ) {
        for(unsigned int j=0; j<4; j++){
            if ( nCorners[j] == contour2f[i] ){
                cornerIndex[j] = i;
                group=j;
            }
        }
        cntPts[group].push_back(contour2f[i]);
    }
    for (int i = 0; i < 4; i++)
    {
        CV_Assert(cornerIndex[i] != -1);
    }
    // saves extra group into corresponding
    if( !cntPts[4].empty() ){
        for( unsigned int i=0; i < cntPts[4].size() ; i++ )
            cntPts[group].push_back(cntPts[4].at(i));
        cntPts[4].clear();
    }

    //Evaluate contour direction :: using the position of the detected corners
    int inc=1;

        inc = ( (cornerIndex[0] > cornerIndex[1]) &&  (cornerIndex[3] > cornerIndex[0]) ) ? -1:inc;
    inc = ( (cornerIndex[2] > cornerIndex[3]) &&  (cornerIndex[1] > cornerIndex[2]) ) ? -1:inc;

    // calculate the line :: who passes through the grouped points
    Point3f lines[4];
    for(int i=0; i<4; i++){
        lines[i]=_interpolate2Dline(cntPts[i]);
    }

    /*
     * calculate the corner :: where the lines crosses to each other
     * clockwise direction        no clockwise direction
     *      0                           1
     *      .---. 1                     .---. 2
     *      |   |                       |   |
     *    3 .___.                     0 .___.
     *          2                           3
     */
    for(int i=0; i < 4; i++){
        if(inc<0)
            nCorners[i] = _getCrossPoint(lines[ i ], lines[ (i+1)%4 ]);    // 01 12 23 30
        else
            nCorners[i] = _getCrossPoint(lines[ i ], lines[ (i+3)%4 ]);    // 30 01 12 23
    }
}

static inline void findCornerInPyrImage(const float scale_init, const int closest_pyr_image_idx,
                                        const vector<Mat>& grey_pyramid, Mat corners,
                                        const DetectorParameters& params) {
    // scale them to the closest pyramid level
    if (scale_init != 1.f)
        corners *= scale_init; // scale_init * scale_pyr
    for (int idx = closest_pyr_image_idx - 1; idx >= 0; --idx) {
        // scale them to new pyramid level
        corners *= 2.f; // *= scale_pyr;
        // use larger win size for larger images
        const int subpix_win_size = std::max(grey_pyramid[idx].cols, grey_pyramid[idx].rows) > 1080 ? 5 : 3;
        cornerSubPix(grey_pyramid[idx], corners,
                     Size(subpix_win_size, subpix_win_size),
                     Size(-1, -1),
                     TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                  params.cornerRefinementMaxIterations,
                                  params.cornerRefinementMinAccuracy));
    }
}

struct ArucoDetector::ArucoDetectorImpl {
    /// dictionary indicates the type of markers that will be searched
    Dictionary dictionary;

    /// marker detection parameters, check DetectorParameters docs to see available settings
    DetectorParameters detectorParams;

    /// marker refine parameters
    RefineParameters refineParams;
    ArucoDetectorImpl() {}

    ArucoDetectorImpl(const Dictionary &_dictionary, const DetectorParameters &_detectorParams,
                      const RefineParameters& _refineParams): dictionary(_dictionary),
                      detectorParams(_detectorParams), refineParams(_refineParams) {}

};

ArucoDetector::ArucoDetector(const Dictionary &_dictionary,
                             const DetectorParameters &_detectorParams,
                             const RefineParameters& _refineParams) {
    arucoDetectorImpl = makePtr<ArucoDetectorImpl>(_dictionary, _detectorParams, _refineParams);
}

void ArucoDetector::detectMarkers(InputArray _image, OutputArrayOfArrays _corners, OutputArray _ids,
                                  OutputArrayOfArrays _rejectedImgPoints) const {
    CV_Assert(!_image.empty());
    DetectorParameters& detectorParams = arucoDetectorImpl->detectorParams;
    const Dictionary& dictionary = arucoDetectorImpl->dictionary;

    CV_Assert(detectorParams.markerBorderBits > 0);
    // check that the parameters are set correctly if Aruco3 is used
    CV_Assert(!(detectorParams.useAruco3Detection == true &&
                detectorParams.minSideLengthCanonicalImg == 0 &&
                detectorParams.minMarkerLengthRatioOriginalImg == 0.0));

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    // Aruco3 functionality is the extension of Aruco.
    // The description can be found in:
    // [1] Speeded up detection of squared fiducial markers, 2018, FJ Romera-Ramirez et al.
    // if Aruco3 functionality if not wanted
    // change some parameters to be sure to turn it off
    if (!detectorParams.useAruco3Detection) {
        detectorParams.minMarkerLengthRatioOriginalImg = 0.0;
        detectorParams.minSideLengthCanonicalImg = 0;
    }
    else {
        // always turn on corner refinement in case of Aruco3, due to upsampling
        detectorParams.cornerRefinementMethod = (int)CORNER_REFINE_SUBPIX;
        // only CORNER_REFINE_SUBPIX implement correctly for useAruco3Detection
        // Todo: update other CORNER_REFINE methods
    }

    /// Step 0: equation (2) from paper [1]
    const float fxfy = (!detectorParams.useAruco3Detection ? 1.f : detectorParams.minSideLengthCanonicalImg /
                       (detectorParams.minSideLengthCanonicalImg + std::max(grey.cols, grey.rows)*
                       detectorParams.minMarkerLengthRatioOriginalImg));

    /// Step 1: create image pyramid. Section 3.4. in [1]
    vector<Mat> grey_pyramid;
    int closest_pyr_image_idx = 0, num_levels = 0;
    //// Step 1.1: resize image with equation (1) from paper [1]
    if (detectorParams.useAruco3Detection) {
        const float scale_pyr = 2.f;
        const float img_area = static_cast<float>(grey.rows*grey.cols);
        const float min_area_marker = static_cast<float>(detectorParams.minSideLengthCanonicalImg*
                                                         detectorParams.minSideLengthCanonicalImg);
        // find max level
        num_levels = static_cast<int>(log2(img_area / min_area_marker)/scale_pyr);
        // the closest pyramid image to the downsampled segmentation image
        // will later be used as start index for corner upsampling
        const float scale_img_area = img_area * fxfy * fxfy;
        closest_pyr_image_idx = cvRound(log2(img_area / scale_img_area)/scale_pyr);
    }
    buildPyramid(grey, grey_pyramid, num_levels);

    // resize to segmentation image
    // in this reduces size the contours will be detected
    if (fxfy != 1.f)
        resize(grey, grey, Size(cvRound(fxfy * grey.cols), cvRound(fxfy * grey.rows)));

    /// STEP 2: Detect marker candidates
    vector<vector<Point2f> > candidates;
    vector<vector<Point> > contours;
    vector<int> ids;

    vector<vector<vector<Point2f> > > candidatesSet;
    vector<vector<vector<Point> > > contoursSet;

    /// STEP 2.a Detect marker candidates :: using AprilTag
    if(detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_APRILTAG){
        _apriltag(grey, detectorParams, candidates, contours);

        candidatesSet.push_back(candidates);
        contoursSet.push_back(contours);
    }
    /// STEP 2.b Detect marker candidates :: traditional way
    else
        _detectCandidates(grey, candidatesSet, contoursSet, detectorParams);

    /// STEP 2: Check candidate codification (identify markers)
    _identifyCandidates(grey, grey_pyramid, candidatesSet, contoursSet, dictionary,
                        candidates, contours, ids, detectorParams, _rejectedImgPoints);

    /// STEP 3: Corner refinement :: use corner subpix
    if (detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_SUBPIX) {
        CV_Assert(detectorParams.cornerRefinementWinSize > 0 && detectorParams.cornerRefinementMaxIterations > 0 &&
                  detectorParams.cornerRefinementMinAccuracy > 0);
        // Do subpixel estimation. In Aruco3 start on the lowest pyramid level and upscale the corners
        parallel_for_(Range(0, (int)candidates.size()), [&](const Range& range) {
            const int begin = range.start;
            const int end = range.end;

            for (int i = begin; i < end; i++) {
                if (detectorParams.useAruco3Detection) {
                    const float scale_init = (float) grey_pyramid[closest_pyr_image_idx].cols / grey.cols;
                    findCornerInPyrImage(scale_init, closest_pyr_image_idx, grey_pyramid, Mat(candidates[i]), detectorParams);
                }
                else
                cornerSubPix(grey, Mat(candidates[i]),
                             Size(detectorParams.cornerRefinementWinSize, detectorParams.cornerRefinementWinSize),
                             Size(-1, -1),
                             TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                          detectorParams.cornerRefinementMaxIterations,
                                          detectorParams.cornerRefinementMinAccuracy));
            }
        });
    }

    /// STEP 3, Optional : Corner refinement :: use contour container
    if (detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_CONTOUR){

        if (!ids.empty()) {

            // do corner refinement using the contours for each detected markers
            parallel_for_(Range(0, (int)candidates.size()), [&](const Range& range) {
                for (int i = range.start; i < range.end; i++) {
                    _refineCandidateLines(contours[i], candidates[i]);
                }
            });
        }
    }

    if (detectorParams.cornerRefinementMethod != (int)CORNER_REFINE_SUBPIX && fxfy != 1.f) {
        // only CORNER_REFINE_SUBPIX implement correctly for useAruco3Detection
        // Todo: update other CORNER_REFINE methods

        // scale to orignal size, this however will lead to inaccurate detections!
        for (auto &vecPoints : candidates)
            for (auto &point : vecPoints)
                point *= 1.f/fxfy;
    }

    // copy to output arrays
    _copyVector2Output(candidates, _corners);
    Mat(ids).copyTo(_ids);
}

/**
  * Project board markers that are not included in the list of detected markers
  */
static inline void _projectUndetectedMarkers(const Board &board, InputOutputArrayOfArrays detectedCorners,
                                             InputOutputArray detectedIds, InputArray cameraMatrix, InputArray distCoeffs,
                                             vector<vector<Point2f> >& undetectedMarkersProjectedCorners,
                                             OutputArray undetectedMarkersIds) {
    Mat rvec, tvec; // first estimate board pose with the current avaible markers
    Mat objPoints, imgPoints; // object and image points for the solvePnP function
    board.matchImagePoints(detectedCorners, detectedIds, objPoints, imgPoints);
    if (objPoints.total() < 4ull) // at least one marker from board so rvec and tvec are valid
        return;
    solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);

    // search undetected markers and project them using the previous pose
    vector<vector<Point2f> > undetectedCorners;
    const std::vector<int>& ids = board.getIds();
    vector<int> undetectedIds;
    for(unsigned int i = 0; i < ids.size(); i++) {
        int foundIdx = -1;
        for(unsigned int j = 0; j < detectedIds.total(); j++) {
            if(ids[i] == detectedIds.getMat().ptr<int>()[j]) {
                foundIdx = j;
                break;
            }
        }

        // not detected
        if(foundIdx == -1) {
            undetectedCorners.push_back(vector<Point2f>());
            undetectedIds.push_back(ids[i]);
            projectPoints(board.getObjPoints()[i], rvec, tvec, cameraMatrix, distCoeffs,
                          undetectedCorners.back());
        }
    }
    // parse output
    Mat(undetectedIds).copyTo(undetectedMarkersIds);
    undetectedMarkersProjectedCorners = undetectedCorners;
}

/**
  * Interpolate board markers that are not included in the list of detected markers using
  * global homography
  */
static void _projectUndetectedMarkers(const Board &_board, InputOutputArrayOfArrays _detectedCorners,
                               InputOutputArray _detectedIds,
                               vector<vector<Point2f> >& _undetectedMarkersProjectedCorners,
                               OutputArray _undetectedMarkersIds) {
    // check board points are in the same plane, if not, global homography cannot be applied
    CV_Assert(_board.getObjPoints().size() > 0);
    CV_Assert(_board.getObjPoints()[0].size() > 0);
    float boardZ = _board.getObjPoints()[0][0].z;
    for(unsigned int i = 0; i < _board.getObjPoints().size(); i++) {
        for(unsigned int j = 0; j < _board.getObjPoints()[i].size(); j++)
            CV_Assert(boardZ == _board.getObjPoints()[i][j].z);
    }

    vector<Point2f> detectedMarkersObj2DAll; // Object coordinates (without Z) of all the detected
                                             // marker corners in a single vector
    vector<Point2f> imageCornersAll; // Image corners of all detected markers in a single vector
    vector<vector<Point2f> > undetectedMarkersObj2D; // Object coordinates (without Z) of all
                                                        // missing markers in different vectors
    vector<int> undetectedMarkersIds; // ids of missing markers
    // find markers included in board, and missing markers from board. Fill the previous vectors
    for(unsigned int j = 0; j < _board.getIds().size(); j++) {
        bool found = false;
        for(unsigned int i = 0; i < _detectedIds.total(); i++) {
            if(_detectedIds.getMat().ptr<int>()[i] == _board.getIds()[j]) {
                for(int c = 0; c < 4; c++) {
                    imageCornersAll.push_back(_detectedCorners.getMat(i).ptr<Point2f>()[c]);
                    detectedMarkersObj2DAll.push_back(
                        Point2f(_board.getObjPoints()[j][c].x, _board.getObjPoints()[j][c].y));
                }
                found = true;
                break;
            }
        }
        if(!found) {
            undetectedMarkersObj2D.push_back(vector<Point2f>());
            for(int c = 0; c < 4; c++) {
                undetectedMarkersObj2D.back().push_back(
                    Point2f(_board.getObjPoints()[j][c].x, _board.getObjPoints()[j][c].y));
            }
            undetectedMarkersIds.push_back(_board.getIds()[j]);
        }
    }
    if(imageCornersAll.size() == 0) return;

    // get homography from detected markers
    Mat transformation = findHomography(detectedMarkersObj2DAll, imageCornersAll);

    _undetectedMarkersProjectedCorners.resize(undetectedMarkersIds.size());

    // for each undetected marker, apply transformation
    for(unsigned int i = 0; i < undetectedMarkersObj2D.size(); i++) {
        perspectiveTransform(undetectedMarkersObj2D[i], _undetectedMarkersProjectedCorners[i], transformation);
    }
    Mat(undetectedMarkersIds).copyTo(_undetectedMarkersIds);
}

void ArucoDetector::refineDetectedMarkers(InputArray _image, const Board& _board,
                                          InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                                          InputOutputArrayOfArrays _rejectedCorners, InputArray _cameraMatrix,
                                          InputArray _distCoeffs, OutputArray _recoveredIdxs) const {
    DetectorParameters& detectorParams = arucoDetectorImpl->detectorParams;
    const Dictionary& dictionary = arucoDetectorImpl->dictionary;
    RefineParameters& refineParams = arucoDetectorImpl->refineParams;
    CV_Assert(refineParams.minRepDistance > 0);

    if(_detectedIds.total() == 0 || _rejectedCorners.total() == 0) return;

    // get projections of missing markers in the board
    vector<vector<Point2f> > undetectedMarkersCorners;
    vector<int> undetectedMarkersIds;
    if(_cameraMatrix.total() != 0) {
        // reproject based on camera projection model
        _projectUndetectedMarkers(_board, _detectedCorners, _detectedIds, _cameraMatrix, _distCoeffs,
                                  undetectedMarkersCorners, undetectedMarkersIds);

    } else {
        // reproject based on global homography
        _projectUndetectedMarkers(_board, _detectedCorners, _detectedIds, undetectedMarkersCorners,
                                  undetectedMarkersIds);
    }

    // list of missing markers indicating if they have been assigned to a candidate
    vector<bool > alreadyIdentified(_rejectedCorners.total(), false);

    // maximum bits that can be corrected
    int maxCorrectionRecalculated =
        int(double(dictionary.maxCorrectionBits) * refineParams.errorCorrectionRate);

    Mat grey;
    _convertToGrey(_image, grey);

    // vector of final detected marker corners and ids
    vector<vector<Point2f> > finalAcceptedCorners;
    vector<int> finalAcceptedIds;
    // fill with the current markers
    finalAcceptedCorners.resize(_detectedCorners.total());
    finalAcceptedIds.resize(_detectedIds.total());
    for(unsigned int i = 0; i < _detectedIds.total(); i++) {
        finalAcceptedCorners[i] = _detectedCorners.getMat(i).clone();
        finalAcceptedIds[i] = _detectedIds.getMat().ptr<int>()[i];
    }
    vector<int> recoveredIdxs; // original indexes of accepted markers in _rejectedCorners

    // for each missing marker, try to find a correspondence
    for(unsigned int i = 0; i < undetectedMarkersIds.size(); i++) {

        // best match at the moment
        int closestCandidateIdx = -1;
        double closestCandidateDistance = refineParams.minRepDistance * refineParams.minRepDistance + 1;
        Mat closestRotatedMarker;

        for(unsigned int j = 0; j < _rejectedCorners.total(); j++) {
            if(alreadyIdentified[j]) continue;

            // check distance
            double minDistance = closestCandidateDistance + 1;
            bool valid = false;
            int validRot = 0;
            for(int c = 0; c < 4; c++) { // first corner in rejected candidate
                double currentMaxDistance = 0;
                for(int k = 0; k < 4; k++) {
                    Point2f rejCorner = _rejectedCorners.getMat(j).ptr<Point2f>()[(c + k) % 4];
                    Point2f distVector = undetectedMarkersCorners[i][k] - rejCorner;
                    double cornerDist = distVector.x * distVector.x + distVector.y * distVector.y;
                    currentMaxDistance = max(currentMaxDistance, cornerDist);
                }
                // if distance is better than current best distance
                if(currentMaxDistance < closestCandidateDistance) {
                    valid = true;
                    validRot = c;
                    minDistance = currentMaxDistance;
                }
                if(!refineParams.checkAllOrders) break;
            }

            if(!valid) continue;

            // apply rotation
            Mat rotatedMarker;
            if(refineParams.checkAllOrders) {
                rotatedMarker = Mat(4, 1, CV_32FC2);
                for(int c = 0; c < 4; c++)
                    rotatedMarker.ptr<Point2f>()[c] =
                        _rejectedCorners.getMat(j).ptr<Point2f>()[(c + 4 + validRot) % 4];
            }
            else rotatedMarker = _rejectedCorners.getMat(j);

            // last filter, check if inner code is close enough to the assigned marker code
            int codeDistance = 0;
            // if errorCorrectionRate, dont check code
            if(refineParams.errorCorrectionRate >= 0) {

                // extract bits
                Mat bits = _extractBits(
                    grey, rotatedMarker, dictionary.markerSize, detectorParams.markerBorderBits,
                    detectorParams.perspectiveRemovePixelPerCell,
                    detectorParams.perspectiveRemoveIgnoredMarginPerCell, detectorParams.minOtsuStdDev);

                Mat onlyBits =
                    bits.rowRange(detectorParams.markerBorderBits, bits.rows - detectorParams.markerBorderBits)
                        .colRange(detectorParams.markerBorderBits, bits.rows - detectorParams.markerBorderBits);

                codeDistance =
                    dictionary.getDistanceToId(onlyBits, undetectedMarkersIds[i], false);
            }

            // if everythin is ok, assign values to current best match
            if(refineParams.errorCorrectionRate < 0 || codeDistance < maxCorrectionRecalculated) {
                closestCandidateIdx = j;
                closestCandidateDistance = minDistance;
                closestRotatedMarker = rotatedMarker;
            }
        }

        // if at least one good match, we have rescue the missing marker
        if(closestCandidateIdx >= 0) {

            // subpixel refinement
            if(detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_SUBPIX) {
                CV_Assert(detectorParams.cornerRefinementWinSize > 0 &&
                          detectorParams.cornerRefinementMaxIterations > 0 &&
                          detectorParams.cornerRefinementMinAccuracy > 0);
                cornerSubPix(grey, closestRotatedMarker,
                             Size(detectorParams.cornerRefinementWinSize, detectorParams.cornerRefinementWinSize),
                             Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                                        detectorParams.cornerRefinementMaxIterations,
                                                        detectorParams.cornerRefinementMinAccuracy));
            }

            // remove from rejected
            alreadyIdentified[closestCandidateIdx] = true;

            // add to detected
            finalAcceptedCorners.push_back(closestRotatedMarker);
            finalAcceptedIds.push_back(undetectedMarkersIds[i]);

            // add the original index of the candidate
            recoveredIdxs.push_back(closestCandidateIdx);
        }
    }

    // parse output
    if(finalAcceptedIds.size() != _detectedIds.total()) {
        // parse output
        Mat(finalAcceptedIds).copyTo(_detectedIds);
        _copyVector2Output(finalAcceptedCorners, _detectedCorners);

        // recalculate _rejectedCorners based on alreadyIdentified
        vector<vector<Point2f> > finalRejected;
        for(unsigned int i = 0; i < alreadyIdentified.size(); i++) {
            if(!alreadyIdentified[i]) {
                finalRejected.push_back(_rejectedCorners.getMat(i).clone());
            }
        }
        _copyVector2Output(finalRejected, _rejectedCorners);

        if(_recoveredIdxs.needed()) {
            Mat(recoveredIdxs).copyTo(_recoveredIdxs);
        }
    }
}

void ArucoDetector::write(FileStorage &fs) const
{
    arucoDetectorImpl->dictionary.writeDictionary(fs);
    arucoDetectorImpl->detectorParams.writeDetectorParameters(fs);
    arucoDetectorImpl->refineParams.writeRefineParameters(fs);
}

void ArucoDetector::read(const FileNode &fn) {
    arucoDetectorImpl->dictionary.readDictionary(fn);
    arucoDetectorImpl->detectorParams.readDetectorParameters(fn);
    arucoDetectorImpl->refineParams.readRefineParameters(fn);
}

const Dictionary& ArucoDetector::getDictionary() const {
    return arucoDetectorImpl->dictionary;
}

void ArucoDetector::setDictionary(const Dictionary& dictionary) {
    arucoDetectorImpl->dictionary = dictionary;
}

const DetectorParameters& ArucoDetector::getDetectorParameters() const {
    return arucoDetectorImpl->detectorParams;
}

void ArucoDetector::setDetectorParameters(const DetectorParameters& detectorParameters) {
    arucoDetectorImpl->detectorParams = detectorParameters;
}

const RefineParameters& ArucoDetector::getRefineParameters() const {
    return arucoDetectorImpl->refineParams;
}

void ArucoDetector::setRefineParameters(const RefineParameters& refineParameters) {
    arucoDetectorImpl->refineParams = refineParameters;
}

void drawDetectedMarkers(InputOutputArray _image, InputArrayOfArrays _corners,
                         InputArray _ids, Scalar borderColor) {
    CV_Assert(_image.getMat().total() != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert((_corners.total() == _ids.total()) || _ids.total() == 0);

    // calculate colors
    Scalar textColor, cornerColor;
    textColor = cornerColor = borderColor;
    swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
    swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

    int nMarkers = (int)_corners.total();
    for(int i = 0; i < nMarkers; i++) {
        Mat currentMarker = _corners.getMat(i);
        CV_Assert(currentMarker.total() == 4 && currentMarker.type() == CV_32FC2);

        // draw marker sides
        for(int j = 0; j < 4; j++) {
            Point2f p0, p1;
            p0 = currentMarker.ptr<Point2f>(0)[j];
            p1 = currentMarker.ptr<Point2f>(0)[(j + 1) % 4];
            line(_image, p0, p1, borderColor, 1);
        }
        // draw first corner mark
        rectangle(_image, currentMarker.ptr<Point2f>(0)[0] - Point2f(3, 3),
                  currentMarker.ptr<Point2f>(0)[0] + Point2f(3, 3), cornerColor, 1, LINE_AA);

        // draw ID
        if(_ids.total() != 0) {
            Point2f cent(0, 0);
            for(int p = 0; p < 4; p++)
                cent += currentMarker.ptr<Point2f>(0)[p];
            cent = cent / 4.;
            stringstream s;
            s << "id=" << _ids.getMat().ptr<int>(0)[i];
            putText(_image, s.str(), cent, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
        }
    }
}

void generateImageMarker(const Dictionary &dictionary, int id, int sidePixels, OutputArray _img, int borderBits) {
    dictionary.generateImageMarker(id, sidePixels, _img, borderBits);
}

}
}
