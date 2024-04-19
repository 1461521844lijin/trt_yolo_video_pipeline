// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_SESSION_HPP
#define GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_SESSION_HPP
#include <stdio.h>
#include <memory>
#include <queue>

#include <opencv2/gapi/streaming/meta.hpp>

#include "streaming/onevpl/engine/engine_session.hpp"
#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
class Surface;
struct VPLAccelerationPolicy;
class VPLLegacyDecodeEngine;

class GAPI_EXPORTS LegacyDecodeSession : public EngineSession {
public:
    friend class VPLLegacyDecodeEngine;
    friend class VPLLegacyTranscodeEngine; //TODO: remove friend add method

    LegacyDecodeSession(mfxSession sess, DecoderParams&& decoder_param, std::shared_ptr<IDataProvider> provider);
    ~LegacyDecodeSession();
    using EngineSession::EngineSession;

    void swap_decode_surface(VPLLegacyDecodeEngine& engine);
    void init_surface_pool(VPLAccelerationPolicy::pool_key_t key);

    Data::Meta generate_frame_meta();
    virtual const mfxFrameInfo& get_video_param() const override;

    IDataProvider::mfx_bitstream *get_mfx_bitstream_ptr();
private:
    mfxVideoParam mfx_decoder_param;
    VPLAccelerationPolicy::pool_key_t decoder_pool_id;

    std::shared_ptr<IDataProvider> data_provider;
    std::shared_ptr<IDataProvider::mfx_bitstream> stream;

protected:
    std::weak_ptr<Surface> processing_surface_ptr;
    using op_handle_t = std::pair<mfxSyncPoint, mfxFrameSurface1*>;
    std::queue<op_handle_t> sync_queue;

    int64_t decoded_frames_count;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONVPL_ENGINE_DECODE_DECODE_SESSION_HPP
