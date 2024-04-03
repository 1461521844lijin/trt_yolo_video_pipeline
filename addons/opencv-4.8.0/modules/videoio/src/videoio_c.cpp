// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/videoio/registry.hpp"
#include "videoio_registry.hpp"

using namespace cv;

// Legacy C-like API

CV_IMPL CvCapture* cvCreateCameraCapture(int)
{
    CV_LOG_WARNING(NULL, "cvCreateCameraCapture doesn't support legacy API anymore.")
    return NULL;
}

CV_IMPL CvCapture* cvCreateFileCaptureWithPreference(const char*, int)
{
    CV_LOG_WARNING(NULL, "cvCreateFileCaptureWithPreference doesn't support legacy API anymore.")
    return NULL;
}

CV_IMPL CvCapture* cvCreateFileCapture(const char * filename)
{
    return cvCreateFileCaptureWithPreference(filename, CAP_ANY);
}

CV_IMPL CvVideoWriter* cvCreateVideoWriter(const char*, int, double, CvSize, int)
{
    CV_LOG_WARNING(NULL, "cvCreateVideoWriter doesn't support legacy API anymore.")
    return NULL;
}

CV_IMPL int cvWriteFrame(CvVideoWriter* writer, const IplImage* image)
{
    return writer ? writer->writeFrame(image) : 0;
}

CV_IMPL void cvReleaseVideoWriter(CvVideoWriter** pwriter)
{
    if( pwriter && *pwriter )
    {
        delete *pwriter;
        *pwriter = 0;
    }
}

CV_IMPL void cvReleaseCapture(CvCapture** pcapture)
{
    if (pcapture && *pcapture)
    {
        delete *pcapture;
        *pcapture = 0;
    }
}

CV_IMPL IplImage* cvQueryFrame(CvCapture* capture)
{
    if (!capture)
        return 0;
    if (!capture->grabFrame())
        return 0;
    return capture->retrieveFrame(0);
}

CV_IMPL int cvGrabFrame(CvCapture* capture)
{
    return capture ? capture->grabFrame() : 0;
}

CV_IMPL IplImage* cvRetrieveFrame(CvCapture* capture, int idx)
{
    return capture ? capture->retrieveFrame(idx) : 0;
}

CV_IMPL double cvGetCaptureProperty(CvCapture* capture, int id)
{
    return capture ? capture->getProperty(id) : 0;
}

CV_IMPL int cvSetCaptureProperty(CvCapture* capture, int id, double value)
{
    return capture ? capture->setProperty(id, value) : 0;
}

CV_IMPL int cvGetCaptureDomain(CvCapture* capture)
{
    return capture ? capture->getCaptureDomain() : 0;
}
