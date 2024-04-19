// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "test_precomp.hpp"
#include "test_aruco_utils.hpp"

namespace opencv_test { namespace {

/**
 * @brief Get a synthetic image of Chessboard in perspective
 */
static Mat projectChessboard(int squaresX, int squaresY, float squareSize, Size imageSize,
                             Mat cameraMatrix, Mat rvec, Mat tvec, bool legacyPattern) {

    Mat img(imageSize, CV_8UC1, Scalar::all(255));
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    for(int y = 0; y < squaresY; y++) {
        float startY = float(y) * squareSize;
        for(int x = 0; x < squaresX; x++) {
            if(legacyPattern && (squaresY % 2 == 0)) {
                if((y + 1) % 2 != x % 2) continue;
            } else {
                if(y % 2 != x % 2) continue;
            }
            float startX = float(x) * squareSize;

            vector< Point3f > squareCorners;
            squareCorners.push_back(Point3f(startX, startY, 0) - Point3f(squaresX*squareSize/2.f, squaresY*squareSize/2.f, 0.f));
            squareCorners.push_back(squareCorners[0] + Point3f(squareSize, 0, 0));
            squareCorners.push_back(squareCorners[0] + Point3f(squareSize, squareSize, 0));
            squareCorners.push_back(squareCorners[0] + Point3f(0, squareSize, 0));

            vector< vector< Point2f > > projectedCorners;
            projectedCorners.push_back(vector< Point2f >());
            projectPoints(squareCorners, rvec, tvec, cameraMatrix, distCoeffs, projectedCorners[0]);

            vector< vector< Point > > projectedCornersInt;
            projectedCornersInt.push_back(vector< Point >());

            for(int k = 0; k < 4; k++)
                projectedCornersInt[0]
                    .push_back(Point((int)projectedCorners[0][k].x, (int)projectedCorners[0][k].y));

            fillPoly(img, projectedCornersInt, Scalar::all(0));
        }
    }

    return img;
}


/**
 * @brief Check pose estimation of charuco board
 */
static Mat projectCharucoBoard(aruco::CharucoBoard& board, Mat cameraMatrix, double yaw,
                               double pitch, double distance, Size imageSize, int markerBorder,
                               Mat &rvec, Mat &tvec) {

    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    // project markers
    Mat img = Mat(imageSize, CV_8UC1, Scalar::all(255));
    for(unsigned int indexMarker = 0; indexMarker < board.getIds().size(); indexMarker++) {
        projectMarker(img, board, indexMarker, cameraMatrix, rvec, tvec, markerBorder);
    }

    // project chessboard
    Mat chessboard =
        projectChessboard(board.getChessboardSize().width, board.getChessboardSize().height,
                          board.getSquareLength(), imageSize, cameraMatrix, rvec, tvec, board.getLegacyPattern());

    for(unsigned int i = 0; i < chessboard.total(); i++) {
        if(chessboard.ptr< unsigned char >()[i] == 0) {
            img.ptr< unsigned char >()[i] = 0;
        }
    }

    return img;
}

/**
 * @brief Check Charuco detection
 */
class CV_CharucoDetection : public cvtest::BaseTest {
    public:
    CV_CharucoDetection(bool _legacyPattern) : legacyPattern(_legacyPattern) {}

    protected:
    void run(int);

    bool legacyPattern;
};


void CV_CharucoDetection::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    aruco::DetectorParameters params;
    params.minDistanceToBorder = 3;
    aruco::CharucoBoard board(Size(4, 4), 0.03f, 0.015f, aruco::getPredefinedDictionary(aruco::DICT_6X6_250));
    board.setLegacyPattern(legacyPattern);
    aruco::CharucoDetector detector(board, aruco::CharucoParameters(), params);

    cameraMatrix.at<double>(0, 0) = cameraMatrix.at<double>(1, 1) = 600;
    cameraMatrix.at<double>(0, 2) = imgSize.width / 2;
    cameraMatrix.at<double>(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    // for different perspectives
    for(double distance : {0.2, 0.4}) {
        for(int yaw = -55; yaw <= 50; yaw += 25) {
            for(int pitch = -55; pitch <= 50; pitch += 25) {

                int markerBorder = iter % 2 + 1;
                iter++;

                // create synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers and interpolate charuco corners
                vector<vector<Point2f> > corners;
                vector<Point2f> charucoCorners;
                vector<int> ids, charucoIds;

                params.markerBorderBits = markerBorder;
                detector.setDetectorParameters(params);

                //detector.detectMarkers(img, corners, ids);
                if(iter % 2 == 0) {
                    detector.detectBoard(img, charucoCorners, charucoIds, corners, ids);
                } else {
                    aruco::CharucoParameters charucoParameters;
                    charucoParameters.cameraMatrix = cameraMatrix;
                    charucoParameters.distCoeffs = distCoeffs;
                    detector.setCharucoParameters(charucoParameters);
                    detector.detectBoard(img, charucoCorners, charucoIds, corners, ids);
                }

                ASSERT_GT(ids.size(), std::vector< int >::size_type(0)) << "Marker detection failed";

                // check results
                vector< Point2f > projectedCharucoCorners;

                // copy chessboardCorners
                vector<Point3f> copyChessboardCorners = board.getChessboardCorners();
                // move copyChessboardCorners points
                for (size_t i = 0; i < copyChessboardCorners.size(); i++)
                    copyChessboardCorners[i] -= board.getRightBottomCorner() / 2.f;
                projectPoints(copyChessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                              projectedCharucoCorners);

                for(unsigned int i = 0; i < charucoIds.size(); i++) {

                    int currentId = charucoIds[i];

                    ASSERT_LT(currentId, (int)board.getChessboardCorners().size()) << "Invalid Charuco corner id";

                    double repError = cv::norm(charucoCorners[i] - projectedCharucoCorners[currentId]);  // TODO cvtest

                    ASSERT_LE(repError, 5.) << "Charuco corner reprojection error too high";
                }
            }
        }
    }
}



/**
 * @brief Check charuco pose estimation
 */
class CV_CharucoPoseEstimation : public cvtest::BaseTest {
    public:
    CV_CharucoPoseEstimation(bool _legacyPattern) : legacyPattern(_legacyPattern) {}

    protected:
    void run(int);

    bool legacyPattern;
};


void CV_CharucoPoseEstimation::run(int) {
    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(750, 750);
    aruco::DetectorParameters params;
    params.minDistanceToBorder = 3;
    aruco::CharucoBoard board(Size(4, 4), 0.03f, 0.015f, aruco::getPredefinedDictionary(aruco::DICT_6X6_250));
    board.setLegacyPattern(legacyPattern);
    aruco::CharucoDetector detector(board, aruco::CharucoParameters(), params);

    cameraMatrix.at<double>(0, 0) = cameraMatrix.at< double >(1, 1) = 1000;
    cameraMatrix.at<double>(0, 2) = imgSize.width / 2;
    cameraMatrix.at<double>(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    // for different perspectives
    for(double distance : {0.2, 0.25}) {
        for(int yaw = -55; yaw <= 50; yaw += 25) {
            for(int pitch = -55; pitch <= 50; pitch += 25) {

                int markerBorder = iter % 2 + 1;
                iter++;

                // get synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers
                vector<vector<Point2f> > corners;
                vector<int> ids;
                params.markerBorderBits = markerBorder;
                detector.setDetectorParameters(params);

                // detect markers and interpolate charuco corners
                vector<Point2f> charucoCorners;
                vector<int> charucoIds;

                if(iter % 2 == 0) {
                    detector.detectBoard(img, charucoCorners, charucoIds, corners, ids);
                } else {
                    aruco::CharucoParameters charucoParameters;
                    charucoParameters.cameraMatrix = cameraMatrix;
                    charucoParameters.distCoeffs = distCoeffs;
                    detector.setCharucoParameters(charucoParameters);
                    detector.detectBoard(img, charucoCorners, charucoIds, corners, ids);
                }
                ASSERT_EQ(ids.size(), board.getIds().size());
                if(charucoIds.size() == 0) continue;

                // estimate charuco pose
                getCharucoBoardPose(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec);


                // check axes
                const float aruco_offset = (board.getSquareLength() - board.getMarkerLength()) / 2.f;
                Point2f offset;
                vector<Point2f> topLeft, bottomLeft;
                if(legacyPattern) { // white box in upper left corner for even row count chessboard patterns
                    offset = Point2f(aruco_offset + board.getSquareLength(), aruco_offset);
                    topLeft = getMarkerById(board.getIds()[1], corners, ids);
                    bottomLeft = getMarkerById(board.getIds()[2], corners, ids);
                } else { // always a black box in the upper left corner
                    offset = Point2f(aruco_offset, aruco_offset);
                    topLeft = getMarkerById(board.getIds()[0], corners, ids);
                    bottomLeft = getMarkerById(board.getIds()[2], corners, ids);
                }
                vector<Point2f> axes = getAxis(cameraMatrix, distCoeffs, rvec, tvec, board.getSquareLength(), offset);
                ASSERT_NEAR(topLeft[0].x, axes[1].x, 3.f);
                ASSERT_NEAR(topLeft[0].y, axes[1].y, 3.f);
                ASSERT_NEAR(bottomLeft[0].x, axes[2].x, 3.f);
                ASSERT_NEAR(bottomLeft[0].y, axes[2].y, 3.f);

                // check estimate result
                vector< Point2f > projectedCharucoCorners;

                projectPoints(board.getChessboardCorners(), rvec, tvec, cameraMatrix, distCoeffs,
                              projectedCharucoCorners);

                for(unsigned int i = 0; i < charucoIds.size(); i++) {

                    int currentId = charucoIds[i];

                    ASSERT_LT(currentId, (int)board.getChessboardCorners().size()) << "Invalid Charuco corner id";

                    double repError = cv::norm(charucoCorners[i] - projectedCharucoCorners[currentId]);  // TODO cvtest

                    ASSERT_LE(repError, 5.) << "Charuco corner reprojection error too high";
                }
            }
        }
    }
}


/**
 * @brief Check diamond detection
 */
class CV_CharucoDiamondDetection : public cvtest::BaseTest {
    public:
    CV_CharucoDiamondDetection();

    protected:
    void run(int);
};


CV_CharucoDiamondDetection::CV_CharucoDiamondDetection() {}


void CV_CharucoDiamondDetection::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    aruco::DetectorParameters params;
    params.minDistanceToBorder = 0;
    float squareLength = 0.03f;
    float markerLength = 0.015f;
    aruco::CharucoBoard board(Size(3, 3), squareLength, markerLength,
                              aruco::getPredefinedDictionary(aruco::DICT_6X6_250));
    aruco::CharucoDetector detector(board);


    cameraMatrix.at<double>(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at<double>(0, 2) = imgSize.width / 2;
    cameraMatrix.at<double>(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    aruco::CharucoParameters charucoParameters;
    charucoParameters.cameraMatrix = cameraMatrix;
    charucoParameters.distCoeffs = distCoeffs;
    detector.setCharucoParameters(charucoParameters);

    // for different perspectives
    for(double distance : {0.2, 0.22}) {
        for(int yaw = -50; yaw <= 50; yaw += 25) {
            for(int pitch = -50; pitch <= 50; pitch += 25) {

                int markerBorder = iter % 2 + 1;
                vector<int> idsTmp;
                for(int i = 0; i < 4; i++)
                    idsTmp.push_back(4 * iter + i);
                board = aruco::CharucoBoard(Size(3, 3), squareLength, markerLength,
                                            aruco::getPredefinedDictionary(aruco::DICT_6X6_250), idsTmp);
                detector.setBoard(board);
                iter++;

                // get synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers
                vector<vector<Point2f>> corners;
                vector<int> ids;
                params.markerBorderBits = markerBorder;
                detector.setDetectorParameters(params);
                //detector.detectMarkers(img, corners, ids);


                // detect diamonds
                vector<vector<Point2f>> diamondCorners;
                vector<Vec4i> diamondIds;

                detector.detectDiamonds(img, diamondCorners, diamondIds, corners, ids);

                // check detect
                if(ids.size() != 4) {
                    ts->printf(cvtest::TS::LOG, "Not enough markers for diamond detection");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                // check results
                if(diamondIds.size() != 1) {
                    ts->printf(cvtest::TS::LOG, "Diamond not detected correctly");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                for(int i = 0; i < 4; i++) {
                    if(diamondIds[0][i] != board.getIds()[i]) {
                        ts->printf(cvtest::TS::LOG, "Incorrect diamond ids");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }


                vector< Point2f > projectedDiamondCorners;

                // copy chessboardCorners
                vector<Point3f> copyChessboardCorners = board.getChessboardCorners();
                // move copyChessboardCorners points
                for (size_t i = 0; i < copyChessboardCorners.size(); i++)
                    copyChessboardCorners[i] -= board.getRightBottomCorner() / 2.f;

                projectPoints(copyChessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                              projectedDiamondCorners);

                vector< Point2f > projectedDiamondCornersReorder(4);
                projectedDiamondCornersReorder[0] = projectedDiamondCorners[0];
                projectedDiamondCornersReorder[1] = projectedDiamondCorners[1];
                projectedDiamondCornersReorder[2] = projectedDiamondCorners[3];
                projectedDiamondCornersReorder[3] = projectedDiamondCorners[2];


                for(unsigned int i = 0; i < 4; i++) {

                    double repError = cv::norm(diamondCorners[0][i] - projectedDiamondCornersReorder[i]);  // TODO cvtest

                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Diamond corner reprojection error too high");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }

                // estimate diamond pose
                vector< Vec3d > estimatedRvec, estimatedTvec;
                getMarkersPoses(diamondCorners, squareLength, cameraMatrix, distCoeffs, estimatedRvec,
                                                 estimatedTvec, noArray(), false);

                // check result
                vector< Point2f > projectedDiamondCornersPose;
                vector< Vec3f > diamondObjPoints(4);
                diamondObjPoints[0] = Vec3f(0.f, 0.f, 0);
                diamondObjPoints[1] = Vec3f(squareLength, 0.f, 0);
                diamondObjPoints[2] = Vec3f(squareLength, squareLength, 0);
                diamondObjPoints[3] = Vec3f(0.f, squareLength, 0);
                projectPoints(diamondObjPoints, estimatedRvec[0], estimatedTvec[0], cameraMatrix,
                              distCoeffs, projectedDiamondCornersPose);

                for(unsigned int i = 0; i < 4; i++) {
                    double repError = cv::norm(projectedDiamondCornersReorder[i] - projectedDiamondCornersPose[i]);  // TODO cvtest

                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Charuco pose error too high");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }
            }
        }
    }
}

/**
* @brief Check charuco board creation
*/
class CV_CharucoBoardCreation : public cvtest::BaseTest {
public:
    CV_CharucoBoardCreation();

protected:
    void run(int);
};

CV_CharucoBoardCreation::CV_CharucoBoardCreation() {}

void CV_CharucoBoardCreation::run(int)
{
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_5X5_250);
    int n = 6;

    float markerSizeFactor = 0.5f;

    for (float squareSize_mm = 5.0f; squareSize_mm < 35.0f; squareSize_mm += 0.1f)
    {
        aruco::CharucoBoard board_meters(Size(n, n), squareSize_mm*1e-3f,
                                         squareSize_mm * markerSizeFactor * 1e-3f, dictionary);

        aruco::CharucoBoard board_millimeters(Size(n, n), squareSize_mm,
                                              squareSize_mm * markerSizeFactor, dictionary);

        for (size_t i = 0; i < board_meters.getNearestMarkerIdx().size(); i++)
        {
            if (board_meters.getNearestMarkerIdx()[i].size() != board_millimeters.getNearestMarkerIdx()[i].size() ||
                board_meters.getNearestMarkerIdx()[i][0] != board_millimeters.getNearestMarkerIdx()[i][0])
            {
                ts->printf(cvtest::TS::LOG,
                    cv::format("Charuco board topology is sensitive to scale with squareSize=%.1f\n",
                        squareSize_mm).c_str());
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                break;
            }
        }
    }
}


TEST(CV_CharucoDetection, accuracy) {
    const bool legacyPattern = false;
    CV_CharucoDetection test(legacyPattern);
    test.safe_run();
}

TEST(CV_CharucoDetection, accuracy_legacyPattern) {
    const bool legacyPattern = true;
    CV_CharucoDetection test(legacyPattern);
    test.safe_run();
}

TEST(CV_CharucoPoseEstimation, accuracy) {
    const bool legacyPattern = false;
    CV_CharucoPoseEstimation test(legacyPattern);
    test.safe_run();
}

TEST(CV_CharucoPoseEstimation, accuracy_legacyPattern) {
    const bool legacyPattern = true;
    CV_CharucoPoseEstimation test(legacyPattern);
    test.safe_run();
}

TEST(CV_CharucoDiamondDetection, accuracy) {
    CV_CharucoDiamondDetection test;
    test.safe_run();
}

TEST(CV_CharucoBoardCreation, accuracy) {
    CV_CharucoBoardCreation test;
    test.safe_run();
}

TEST(Charuco, testCharucoCornersCollinear_true)
{
    int squaresX = 13;
    int squaresY = 28;
    float squareLength = 300;
    float markerLength = 150;
    int dictionaryId = 11;

    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::PredefinedDictionaryType(dictionaryId));

    aruco::CharucoBoard charucoBoard(Size(squaresX, squaresY), squareLength, markerLength, dictionary);

    // consistency with C++98
    const int arrLine[9] = {192, 204, 216, 228, 240, 252, 264, 276, 288};
    vector<int> charucoIdsAxisLine(9, 0);

    for (int i = 0; i < 9; i++){
        charucoIdsAxisLine[i] = arrLine[i];
    }

    const int arrDiag[7] = {198, 209, 220, 231, 242, 253, 264};

    vector<int> charucoIdsDiagonalLine(7, 0);

    for (int i = 0; i < 7; i++){
        charucoIdsDiagonalLine[i] = arrDiag[i];
    }

    bool resultAxisLine = charucoBoard.checkCharucoCornersCollinear(charucoIdsAxisLine);
    EXPECT_TRUE(resultAxisLine);

    bool resultDiagonalLine = charucoBoard.checkCharucoCornersCollinear(charucoIdsDiagonalLine);
    EXPECT_TRUE(resultDiagonalLine);
}

TEST(Charuco, testCharucoCornersCollinear_false)
{
    int squaresX = 13;
    int squaresY = 28;
    float squareLength = 300;
    float markerLength = 150;
    int dictionaryId = 11;

    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::PredefinedDictionaryType(dictionaryId));

    aruco::CharucoBoard charucoBoard(Size(squaresX, squaresY), squareLength, markerLength, dictionary);

    // consistency with C++98
    const int arr[63] = {192, 193, 194, 195, 196, 197, 198, 204, 205, 206, 207, 208,
                                209, 210, 216, 217, 218, 219, 220, 221, 222, 228, 229, 230,
                                231, 232, 233, 234, 240, 241, 242, 243, 244, 245, 246, 252,
                                253, 254, 255, 256, 257, 258, 264, 265, 266, 267, 268, 269,
                                270, 276, 277, 278, 279, 280, 281, 282, 288, 289, 290, 291,
                                292, 293, 294};

    vector<int> charucoIds(63, 0);
    for (int i = 0; i < 63; i++){
        charucoIds[i] = arr[i];
    }

    bool result = charucoBoard.checkCharucoCornersCollinear(charucoIds);

    EXPECT_FALSE(result);
}

// test that ChArUco board detection is subpixel accurate
TEST(Charuco, testBoardSubpixelCoords)
{
    cv::Size res{500, 500};
    cv::Mat K = (cv::Mat_<double>(3,3) <<
        0.5*res.width, 0, 0.5*res.width,
        0, 0.5*res.height, 0.5*res.height,
        0, 0, 1);

    // set expected_corners values
    cv::Mat expected_corners = (cv::Mat_<float>(9,2) <<
        200, 200,
        250, 200,
        300, 200,
        200, 250,
        250, 250,
        300, 250,
        200, 300,
        250, 300,
        300, 300
    );

    cv::Mat gray;

    aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    aruco::CharucoBoard board(Size(4, 4), 1.f, .8f, dict);

    // generate ChArUco board
    board.generateImage(Size(res.width, res.height), gray, 150);
    cv::GaussianBlur(gray, gray, Size(5, 5), 1.0);

    aruco::DetectorParameters params;
    params.cornerRefinementMethod = (int)cv::aruco::CORNER_REFINE_APRILTAG;

    aruco::CharucoParameters charucoParameters;
    charucoParameters.cameraMatrix = K;
    aruco::CharucoDetector detector(board, charucoParameters);
    detector.setDetectorParameters(params);

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::Mat c_ids, c_corners;

    detector.detectBoard(gray, c_corners, c_ids, corners, ids);

    ASSERT_EQ(ids.size(), size_t(8));
    ASSERT_EQ(c_corners.rows, expected_corners.rows);
    EXPECT_NEAR(0, cvtest::norm(expected_corners, c_corners.reshape(1), NORM_INF), 1e-1);
}

TEST(Charuco, issue_14014)
{
    string imgPath = cvtest::findDataFile("aruco/recover.png");
    Mat img = imread(imgPath);

    aruco::DetectorParameters detectorParams;
    detectorParams.cornerRefinementMethod = (int)aruco::CORNER_REFINE_SUBPIX;
    detectorParams.cornerRefinementMinAccuracy = 0.01;
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_7X7_250), detectorParams);
    aruco::CharucoBoard board(Size(8, 5), 0.03455f, 0.02164f, detector.getDictionary());

    vector<Mat> corners, rejectedPoints;
    vector<int> ids;

    detector.detectMarkers(img, corners, ids, rejectedPoints);

    ASSERT_EQ(corners.size(), 19ull);
    EXPECT_EQ(Size(4, 1), corners[0].size()); // check dimension of detected corners

    size_t numRejPoints = rejectedPoints.size();
    ASSERT_EQ(rejectedPoints.size(), 26ull); // optional check to track regressions
    EXPECT_EQ(Size(4, 1), rejectedPoints[0].size()); // check dimension of detected corners

    detector.refineDetectedMarkers(img, board, corners, ids, rejectedPoints);

    ASSERT_EQ(corners.size(), 20ull);
    EXPECT_EQ(Size(4, 1), corners[0].size()); // check dimension of rejected corners after successfully refine

    ASSERT_EQ(rejectedPoints.size() + 1, numRejPoints);
    EXPECT_EQ(Size(4, 1), rejectedPoints[0].size()); // check dimension of rejected corners after successfully refine
}


TEST(Charuco, testmatchImagePoints)
{
    aruco::CharucoBoard board(Size(2, 3), 1.f, 0.5f, aruco::getPredefinedDictionary(aruco::DICT_4X4_50));
    auto chessboardPoints = board.getChessboardCorners();

    vector<int> detectedIds;
    vector<Point2f> detectedCharucoCorners;
    for (const Point3f& point : chessboardPoints) {
        detectedIds.push_back((int)detectedCharucoCorners.size());
        detectedCharucoCorners.push_back({2.f*point.x, 2.f*point.y});
    }

    vector<Point3f> objPoints;
    vector<Point2f> imagePoints;
    board.matchImagePoints(detectedCharucoCorners, detectedIds, objPoints, imagePoints);

    ASSERT_EQ(detectedCharucoCorners.size(), objPoints.size());
    ASSERT_EQ(detectedCharucoCorners.size(), imagePoints.size());

    for (size_t i = 0ull; i < detectedCharucoCorners.size(); i++) {
        EXPECT_EQ(detectedCharucoCorners[i], imagePoints[i]);
        EXPECT_EQ(chessboardPoints[i].x, objPoints[i].x);
        EXPECT_EQ(chessboardPoints[i].y, objPoints[i].y);
    }
}

typedef testing::TestWithParam<cv::Size> CharucoBoard;
INSTANTIATE_TEST_CASE_P(/**/, CharucoBoard, testing::Values(Size(3, 2), Size(3, 2), Size(6, 2), Size(2, 6),
                                                            Size(3, 4), Size(4, 3), Size(7, 3), Size(3, 7)));
TEST_P(CharucoBoard, testWrongSizeDetection)
{
    cv::Size boardSize = GetParam();
    ASSERT_FALSE(boardSize.width == boardSize.height);
    aruco::CharucoBoard board(boardSize, 1.f, 0.5f, aruco::getPredefinedDictionary(aruco::DICT_4X4_50));

    vector<int> detectedCharucoIds, detectedArucoIds;
    vector<Point2f> detectedCharucoCorners;
    vector<vector<Point2f>> detectedArucoCorners;
    Mat boardImage;
    board.generateImage(boardSize*40, boardImage);

    swap(boardSize.width, boardSize.height);
    aruco::CharucoDetector detector(aruco::CharucoBoard(boardSize, 1.f, 0.5f, aruco::getPredefinedDictionary(aruco::DICT_4X4_50)));
    // try detect board with wrong size
    detector.detectBoard(boardImage, detectedCharucoCorners, detectedCharucoIds, detectedArucoCorners, detectedArucoIds);

    // aruco markers must be found
    ASSERT_EQ(detectedArucoIds.size(), board.getIds().size());
    ASSERT_EQ(detectedArucoCorners.size(), board.getIds().size());
    // charuco corners should not be found in board with wrong size
    ASSERT_TRUE(detectedCharucoCorners.empty());
    ASSERT_TRUE(detectedCharucoIds.empty());
}

}} // namespace
