#pragma once
#include "Demuxer.h"
#include "SafeAVFormat.h"

namespace FFmpeg {
class Decoder {
public:
    Decoder(std::shared_ptr<Demuxer> demux);

    ~Decoder();

    bool open(bool use_hw = false);

    bool send(av_packet packet);

    bool receive(av_frame &frame);

    void close();

    static std::shared_ptr<Decoder> createShare(std::shared_ptr<Demuxer> demux) {
        return std::make_shared<Decoder>(demux);
    }

private:
    av_codec_context         m_codec_ctx;
    std::shared_ptr<Demuxer> m_demux;
    // log
    //std::shared_ptr<spdlog::logger> logger = nullptr;
};
}  // namespace FFmpeg
