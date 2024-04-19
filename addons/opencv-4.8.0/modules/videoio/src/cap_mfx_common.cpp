// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "cap_mfx_common.hpp"

// Linux specific
#ifdef __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

using namespace std;
using namespace cv;

#ifndef HAVE_ONEVPL
static mfxIMPL getImpl()
{
    static const size_t res = utils::getConfigurationParameterSizeT("OPENCV_VIDEOIO_MFX_IMPL", MFX_IMPL_AUTO_ANY);
    return (mfxIMPL)res;
}
#endif

static size_t getExtraSurfaceNum()
{
    static const size_t res = cv::utils::getConfigurationParameterSizeT("OPENCV_VIDEOIO_MFX_EXTRA_SURFACE_NUM", 1);
    return res;
}

static size_t getPoolTimeoutSec()
{
    static const size_t res = utils::getConfigurationParameterSizeT("OPENCV_VIDEOIO_MFX_POOL_TIMEOUT", 1);
    return res;
}

#ifdef HAVE_ONEVPL
// oneVPL loader singleton (HW implementation only)
static mfxLoader setupVPLLoader()
{
    mfxLoader instance = MFXLoad();
    mfxConfig cfg = MFXCreateConfig(instance);
    mfxVariant impl;
    impl.Type = MFX_VARIANT_TYPE_U32;
    impl.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    MFXSetConfigFilterProperty(cfg, (const mfxU8*)"mfxImplDescription.Impl", impl);
    DBG(cerr << "MFX Load: " << instance << endl);
    return instance;
}

mfxLoader getVPLLoaderInstance()
{
    static mfxLoader instance = setupVPLLoader();
    return instance;
}
#endif

//==================================================================================================

bool DeviceHandler::init(MFXVideoSession_WRAP &session)
{
    mfxStatus res = MFX_ERR_NONE;
    mfxVersion ver = { {19, 1} };

#ifdef HAVE_ONEVPL
    res = session.CreateSession();
    DBG(cout << "MFX CreateSession: " << res << endl);
#else
    mfxIMPL impl = getImpl();

    res = session.Init(impl, &ver);
    DBG(cout << "MFX SessionInit: " << res << endl);

    res = session.QueryIMPL(&impl);
    DBG(cout << "MFX QueryIMPL: " << res << " => " << asHex(impl) << endl);
#endif

    res = session.QueryVersion(&ver);
    DBG(cout << "MFX QueryVersion: " << res << " => " << ver.Major << "." << ver.Minor << endl);

    if (res != MFX_ERR_NONE)
        return false;

    return initDeviceSession(session);
}

//==================================================================================================

#ifdef __linux__

VAHandle::VAHandle() {
    // TODO: provide a way of modifying this path
    const string filename = "/dev/dri/renderD128";
    file = open(filename.c_str(), O_RDWR);
    if (file < 0)
        CV_Error(Error::StsError, "Can't open file: " + filename);
    display = vaGetDisplayDRM(file);
}

VAHandle::~VAHandle() {
    if (display) {
        vaTerminate(display);
    }
    if (file >= 0) {
        close(file);
    }
}

bool VAHandle::initDeviceSession(MFXVideoSession_WRAP &session) {
    int majorVer = 0, minorVer = 0;
    VAStatus va_res = vaInitialize(display, &majorVer, &minorVer);
    DBG(cout << "vaInitialize: " << va_res << endl << majorVer << '.' << minorVer << endl);
    if (va_res == VA_STATUS_SUCCESS) {
        mfxStatus mfx_res = session.SetHandle(static_cast<mfxHandleType>(MFX_HANDLE_VA_DISPLAY), display);
        DBG(cout << "MFX SetHandle: " << mfx_res << endl);
        if (mfx_res == MFX_ERR_NONE) {
            return true;
        }
    }
    return false;
}

#endif // __linux__

DeviceHandler * createDeviceHandler()
{
#if defined __linux__
    return new VAHandle();
#elif defined _WIN32
    return new DXHandle();
#else
    return 0;
#endif
}

//==================================================================================================

SurfacePool::SurfacePool(ushort width_, ushort height_, ushort count, const mfxFrameInfo &frameInfo, uchar bpp)
    : width(alignSize(width_, 32)),
      height(alignSize(height_, 32)),
      oneSize(width * height * bpp / 8),
      buffers(count * oneSize),
      surfaces(count)
{
    for(int i = 0; i < count; ++i)
    {
        mfxFrameSurface1 &surface = surfaces[i];
        uint8_t * dataPtr = buffers.data() + oneSize * i;
        memset(&surface, 0, sizeof(mfxFrameSurface1));
        surface.Info = frameInfo;
        surface.Data.Y = dataPtr;
        surface.Data.UV = dataPtr + width * height;
        surface.Data.PitchLow = width & 0xFFFF;
        surface.Data.PitchHigh = (width >> 16) & 0xFFFF;
        DBG(cout << "allocate surface " << (void*)&surface << ", Y = " << (void*)dataPtr << " (" << width << "x" << height << ")" << endl);
    }
    DBG(cout << "Allocated: " << endl
         << "- surface data: " << buffers.size() << " bytes" << endl
         << "- surface headers: " << surfaces.size() * sizeof(mfxFrameSurface1) << " bytes" << endl);
}

SurfacePool::~SurfacePool()
{
}

SurfacePool * SurfacePool::_create(const mfxFrameAllocRequest &request, const mfxVideoParam &params)
{
    return new SurfacePool(request.Info.Width,
                           request.Info.Height,
                           saturate_cast<ushort>((size_t)request.NumFrameSuggested + getExtraSurfaceNum()),
                           params.mfx.FrameInfo);
}

mfxFrameSurface1 *SurfacePool::getFreeSurface()
{
    const int64 start = cv::getTickCount();
    do
    {
        for(std::vector<mfxFrameSurface1>::iterator i = surfaces.begin(); i != surfaces.end(); ++i)
            if (!i->Data.Locked)
                return &(*i);
        sleep_ms(10);
    }
    while((cv::getTickCount() - start) / cv::getTickFrequency() < getPoolTimeoutSec()); // seconds
    DBG(cout << "No free surface!" << std::endl);
    return 0;
}

//==================================================================================================

ReadBitstream::ReadBitstream(const char *filename, size_t maxSize) : drain(false)
{
    input.open(filename, std::ios::in | std::ios::binary);
    DBG(cout << "Open " << filename << " -> " << input.is_open() << std::endl);
    memset(&stream, 0, sizeof(stream));
    stream.MaxLength = (mfxU32)maxSize;
    stream.Data = new mfxU8[stream.MaxLength];
    CV_Assert(stream.Data);
}

ReadBitstream::~ReadBitstream()
{
    delete[] stream.Data;
}

bool ReadBitstream::isOpened() const
{
    return input.is_open();
}

bool ReadBitstream::isDone() const
{
    return input.eof();
}

bool ReadBitstream::read()
{
    memmove(stream.Data, stream.Data + stream.DataOffset, stream.DataLength);
    stream.DataOffset = 0;
    input.read((char*)(stream.Data + stream.DataLength), stream.MaxLength - stream.DataLength);
    if (input.eof() || input.good())
    {
        mfxU32 bytesRead = (mfxU32)input.gcount();
        if (bytesRead > 0)
        {
            stream.DataLength += bytesRead;
            DBG(cout << "read " << bytesRead << " bytes" << endl);
            return true;
        }
    }
    return false;
}

//==================================================================================================

WriteBitstream::WriteBitstream(const char * filename, size_t maxSize)
{
    output.open(filename, std::ios::out | std::ios::binary);
    DBG(cout << "BS Open " << filename << " -> " << output.is_open() << std::endl);
    memset(&stream, 0, sizeof(stream));
    stream.MaxLength = (mfxU32)maxSize;
    stream.Data = new mfxU8[stream.MaxLength];
    DBG(cout << "BS Allocate " << maxSize << " bytes (" << ((float)maxSize / (1 << 20)) << " Mb)" << endl);
    CV_Assert(stream.Data);
}

WriteBitstream::~WriteBitstream()
{
    delete[] stream.Data;
}

bool WriteBitstream::write()
{
    output.write((char*)(stream.Data + stream.DataOffset), stream.DataLength);
    stream.DataLength = 0;
    return output.good();
}

bool WriteBitstream::isOpened() const
{
    return output.is_open();
}
