// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CAP_MFX_HPP
#define CAP_MFX_HPP

#include "precomp.hpp"


class MFXVideoSession_WRAP;
class Plugin;
class DeviceHandler;
class ReadBitstream;
class SurfacePool;
class MFXVideoDECODE;

class VideoCapture_IntelMFX : public cv::IVideoCapture
{
public:
    VideoCapture_IntelMFX(const cv::String &filename);
    ~VideoCapture_IntelMFX();
    double getProperty(int) const CV_OVERRIDE;
    bool setProperty(int, double) CV_OVERRIDE;
    bool grabFrame() CV_OVERRIDE;
    bool retrieveFrame(int, cv::OutputArray out) CV_OVERRIDE;
    bool isOpened() const CV_OVERRIDE;
    int getCaptureDomain() CV_OVERRIDE;
private:
    MFXVideoSession_WRAP *session;
    Plugin *plugin;
    DeviceHandler *deviceHandler;
    ReadBitstream *bs;
    MFXVideoDECODE *decoder;
    SurfacePool *pool;
    void *outSurface;
    cv::Size frameSize;
    bool good;
};


#endif
