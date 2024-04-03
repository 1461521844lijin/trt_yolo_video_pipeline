// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "cap_mfx_reader.hpp"
#include "opencv2/core/base.hpp"
#include "cap_mfx_common.hpp"
#include "opencv2/imgproc/hal/hal.hpp"
#include "cap_interface.hpp"

using namespace cv;
using namespace std;

inline bool hasExtension(const String &filename, const String &ext)
{
    if (filename.size() <= ext.size())
        return false;
    const size_t diff = filename.size() - ext.size();
    const size_t found_at = filename.rfind(ext);
    return found_at == diff;
}

inline mfxU32 determineCodecId(const String &filename)
{
    if (hasExtension(filename, ".h264") || hasExtension(filename, ".264"))
        return MFX_CODEC_AVC;
    else if (hasExtension(filename, ".mp2") || hasExtension(filename, ".mpeg2"))
        return MFX_CODEC_MPEG2;
    else if (hasExtension(filename, ".265") || hasExtension(filename, ".hevc"))
        return MFX_CODEC_HEVC;
    else
        return (mfxU32)-1;
}

//==========================================================================

VideoCapture_IntelMFX::VideoCapture_IntelMFX(const cv::String &filename)
    : session(0), plugin(0), deviceHandler(0), bs(0), decoder(0), pool(0), outSurface(0), good(false)
{
    mfxStatus res = MFX_ERR_NONE;

    // Init device and session
    deviceHandler = createDeviceHandler();
    session = new MFXVideoSession_WRAP();
    if (!deviceHandler->init(*session))
    {
        MSG(cerr << "MFX: Can't initialize session" << endl);
        return;
    }

    // Load appropriate plugin

    mfxU32 codecId = determineCodecId(filename);
    if (codecId == (mfxU32)-1)
    {
        MSG(cerr << "MFX: Unsupported extension: " << filename << endl);
        return;
    }
    plugin = Plugin::loadDecoderPlugin(*session, codecId);
    if (plugin && !plugin->isGood())
    {
        MSG(cerr << "MFX: LoadPlugin failed for codec: " << codecId << " (" << filename << ")" << endl);
        return;
    }

    // Read some content from file

    bs = new ReadBitstream(filename.c_str());
    if (!bs->read())
    {
        MSG(cerr << "MFX: Failed to read bitstream" << endl);
        return;
    }

    // Create decoder and decode stream header

    decoder = new MFXVideoDECODE(*session);
    mfxVideoParam params;
    memset(&params, 0, sizeof(params));
    params.mfx.CodecId = codecId;
    params.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    res = decoder->DecodeHeader(&bs->stream, &params);
    DBG(cout << "DecodeHeader: " << res << endl << params.mfx << params.mfx.FrameInfo << endl);
    if (res < MFX_ERR_NONE)
    {
        MSG(cerr << "MFX: Failed to decode stream header: " << res << endl);
        return;
    }

    // Adjust parameters - COMMENTED: h265 decoder resets crop size to 0 (oneVPL/Win)

    //res = decoder->Query(&params, &params);
    //DBG(cout << "MFX Query: " << res << endl << params.mfx << params.mfx.FrameInfo);
    //CV_Assert(res >= MFX_ERR_NONE);

    // Init surface pool

    pool = SurfacePool::create(decoder, params);
    if (!pool)
    {
        MSG(cerr << "MFX: Failed to create surface pool" << endl);
        return;
    }

    // Init decoder

    res = decoder->Init(&params);
    DBG(cout << "MFX decoder Init: " << res << endl << params.mfx.FrameInfo);
    if (res < MFX_ERR_NONE)
    {
        MSG(cerr << "MFX: Failed to init decoder: " << res << endl);
        return;
    }

    frameSize = Size(params.mfx.FrameInfo.CropW, params.mfx.FrameInfo.CropH);
    if (frameSize == Size(0, 0)) // sometimes Crop size is 0
    {
        frameSize = Size(params.mfx.FrameInfo.Width, params.mfx.FrameInfo.Height);
    }
    good = true;
}


VideoCapture_IntelMFX::~VideoCapture_IntelMFX()
{
    cleanup(plugin);
    cleanup(bs);
    cleanup(decoder);
    cleanup(pool);
    session->Close();
    cleanup(session);
    cleanup(deviceHandler);
}

double VideoCapture_IntelMFX::getProperty(int prop) const
{
    if (!good)
    {
        MSG(cerr << "MFX: can not call getProperty(), backend has not been initialized" << endl);
        return 0;
    }
    switch (prop)
    {
        case CAP_PROP_FRAME_WIDTH:
            return frameSize.width;
        case CAP_PROP_FRAME_HEIGHT:
            return frameSize.height;
        default:
            MSG(cerr << "MFX: unsupported property" << endl);
            return 0;
    }
}

bool VideoCapture_IntelMFX::setProperty(int, double)
{
    MSG(cerr << "MFX: setProperty() is not implemented" << endl);
    return false;
}

bool VideoCapture_IntelMFX::grabFrame()
{
    mfxStatus res;
    mfxFrameSurface1 *workSurface = 0;
    mfxSyncPoint sync;

    workSurface = pool->getFreeSurface();

    while (true)
    {
        if (!workSurface)
        {
            // not enough surfaces
            MSG(cerr << "MFX: Failed to get free surface" << endl);
            return false;
        }

        outSurface = 0;
        res = decoder->DecodeFrameAsync(bs->drain ? 0 : &bs->stream, workSurface, (mfxFrameSurface1**)&outSurface, &sync);
        if (res == MFX_ERR_NONE)
        {
            res = session->SyncOperation(sync, 1000); // 1 sec, TODO: provide interface to modify timeout
            if (res == MFX_ERR_NONE)
            {
                // ready to retrieve
                DBG(cout << "Frame ready to retrieve" << endl);
                return true;
            }
            else
            {
                MSG(cerr << "MFX: Sync error: " << res << endl);
                return false;
            }
        }
        else if (res == MFX_ERR_MORE_DATA)
        {
            if (bs->isDone())
            {
                if (bs->drain)
                {
                    // finish
                    DBG(cout << "Drain finished" << endl);
                    return false;
                }
                else
                {
                    DBG(cout << "Bitstream finished - Drain started" << endl);
                    bs->drain = true;
                    continue;
                }
            }
            else
            {
                bool read_res = bs->read();
                if (!read_res)
                {
                    // failed to read
                    MSG(cerr << "MFX: Bitstream read failure" << endl);
                    return false;
                }
                else
                {
                    DBG(cout << "Bitstream read success" << endl);
                    continue;
                }
            }
        }
        else if (res == MFX_ERR_MORE_SURFACE)
        {
            DBG(cout << "Getting another surface" << endl);
            workSurface = pool->getFreeSurface();
            continue;
        }
        else if (res == MFX_WRN_DEVICE_BUSY)
        {
            DBG(cout << "Waiting for device" << endl);
            sleep_ms(1000);
            continue;
        }
        else if (res == MFX_WRN_VIDEO_PARAM_CHANGED)
        {
            DBG(cout << "Video param changed" << endl);
            continue;
        }
        else
        {
            MSG(cerr << "MFX: Bad status: " << res << endl);
            return false;
        }
    }
}


bool VideoCapture_IntelMFX::retrieveFrame(int, OutputArray out)
{
    if (!outSurface)
    {
        MSG(cerr << "MFX: No frame ready to retrieve" << endl);
        return false;
    }
    mfxFrameSurface1 * s = (mfxFrameSurface1*)outSurface;
    mfxFrameInfo &info = s->Info;
    mfxFrameData &data = s->Data;

    const int cols = info.CropW;
    const int rows = info.CropH;

    out.create(rows, cols, CV_8UC3);
    Mat res = out.getMat();

    hal::cvtTwoPlaneYUVtoBGR(data.Y, data.UV, data.Pitch, res.data, res.step, cols, rows, 3, false, 0);

    return true;
}

bool VideoCapture_IntelMFX::isOpened() const
{
    return good;
}

int VideoCapture_IntelMFX::getCaptureDomain()
{
    return CAP_INTEL_MFX;
}

//==================================================================================================

cv::Ptr<IVideoCapture> cv::create_MFX_capture(const std::string &filename)
{
    return cv::makePtr<VideoCapture_IntelMFX>(filename);
}
