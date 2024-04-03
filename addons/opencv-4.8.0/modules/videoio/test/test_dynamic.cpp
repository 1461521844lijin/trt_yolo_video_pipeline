// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test { namespace {

const int FRAME_COUNT = 120;

inline void generateFrame(int i, Mat & frame)
{
    ::generateFrame(i, FRAME_COUNT, frame);
}

TEST(videoio_dynamic, basic_write)
{
    const Size FRAME_SIZE(640, 480);
    const double FPS = 100;
    const String filename = cv::tempfile(".avi");
    const int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');

    bool fileExists = false;
    {
        vector<VideoCaptureAPIs> backends = videoio_registry::getWriterBackends();
        for (VideoCaptureAPIs be : backends)
        {
            VideoWriter writer;
            writer.open(filename, be, fourcc, FPS, FRAME_SIZE, true);
            if (writer.isOpened())
            {
                Mat frame(FRAME_SIZE, CV_8UC3);
                for (int j = 0; j < FRAME_COUNT; ++j)
                {
                    generateFrame(j, frame);
                    writer << frame;
                }
                writer.release();
                fileExists = true;
            }
            EXPECT_FALSE(writer.isOpened());
        }
    }
    if (!fileExists)
    {
        cout << "None of backends has been able to write video file - SKIP reading part" << endl;
        return;
    }
    {
        vector<VideoCaptureAPIs> backends = videoio_registry::getStreamBackends();
        for (VideoCaptureAPIs be : backends)
        {
            VideoCapture cap;
            cap.open(filename, be);
            if(cap.isOpened())
            {
                int count = 0;
                while (true)
                {
                    Mat frame;
                    if (cap.grab())
                    {
                        if (cap.retrieve(frame))
                        {
                            ++count;
                            continue;
                        }
                    }
                    break;
                }
                EXPECT_EQ(count, FRAME_COUNT);
                cap.release();
            }
            EXPECT_FALSE(cap.isOpened());
        }
    }
    remove(filename.c_str());
}

TEST(videoio_dynamic, write_invalid)
{
    vector<VideoCaptureAPIs> backends = videoio_registry::getWriterBackends();
    for (VideoCaptureAPIs be : backends)
    {
        SCOPED_TRACE(be);
        const string filename = cv::tempfile(".mkv");
        VideoWriter writer;
        bool res = true;

        // Bad FourCC
        EXPECT_NO_THROW(res = writer.open(filename, be, VideoWriter::fourcc('A', 'B', 'C', 'D'), 1, Size(640, 480), true));
        EXPECT_FALSE(res);
        EXPECT_FALSE(writer.isOpened());

        // Empty filename
        EXPECT_NO_THROW(res = writer.open(String(), be, VideoWriter::fourcc('H', '2', '6', '4'), 1, Size(640, 480), true));
        EXPECT_FALSE(res);
        EXPECT_FALSE(writer.isOpened());
        EXPECT_NO_THROW(res = writer.open(String(), be, VideoWriter::fourcc('M', 'J', 'P', 'G'), 1, Size(640, 480), true));
        EXPECT_FALSE(res);
        EXPECT_FALSE(writer.isOpened());

        // zero FPS
        EXPECT_NO_THROW(res = writer.open(filename, be, VideoWriter::fourcc('H', '2', '6', '4'), 0, Size(640, 480), true));
        EXPECT_FALSE(res);
        EXPECT_FALSE(writer.isOpened());

        // cleanup
        EXPECT_NO_THROW(writer.release());
        remove(filename.c_str());
    }

    // Generic
    {
        VideoWriter writer;
        bool res = true;
        EXPECT_NO_THROW(res = writer.open(std::string(), VideoWriter::fourcc('H', '2', '6', '4'), 1, Size(640, 480)));
        EXPECT_FALSE(res);
        EXPECT_FALSE(writer.isOpened());
        EXPECT_NO_THROW(res = writer.open(std::string(), VideoWriter::fourcc('M', 'J', 'P', 'G'), 1, Size(640, 480)));
        EXPECT_FALSE(res);
        EXPECT_FALSE(writer.isOpened());
    }
}


}} // opencv_test::<anonymous>::
