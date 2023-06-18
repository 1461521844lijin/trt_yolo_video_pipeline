#include "StreamPusher.h"

extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

namespace FFmpeg {

    void StreamPusher::worker() {
#define ASSERT(X, STR)                               \
    if (X == nullptr)                                \
    {                                                \
        std::cout << STR << std::endl;               \
        continue;                                    \
    }

        std::shared_ptr<Data::Input_Data> data;
        while (m_run) {
            for (auto &input: m_input_buffers) {
                if (input.second->Pop(data)) {
                    std::shared_ptr<Data::Decode_Data> data_ptr =
                            std::dynamic_pointer_cast<Data::Decode_Data>(data);
                    auto frame = m_scale.Scale(data_ptr->original_image);
                    ASSERT(frame, "scale frame is null");
                    auto pkt = m_encode.Encode(frame.get());
                    ASSERT(pkt, "encode pkt is null");
                    m_mux.Write(pkt);
                } else {
                    if (!m_run)
                        break;
                    std::unique_lock<std::mutex> lk(m_base_mutex);
                    m_base_cond->wait(lk);
                }
            }
        }
    }

    void StreamPusher::Open(const std::string url, bool useHwEncode) {
        if (from_width == 0 || to_width == 0) {
            // LOG(ERROR) << "from_width or to_width is 0";
            return;
        }
        m_scale.InitScale(from_width, from_height, from_format, to_width, to_height, to_format);
        AVCodecContext *codecconetxt = nullptr;
        if (!useHwEncode) {
            codecconetxt = m_encode.Create(AV_CODEC_ID_H264, true);
        } else {
            codecconetxt = m_encode.CreateHwEncode("h264_nvenc", to_width, to_height);
        }

        m_encode.set_c(codecconetxt);
        m_encode.SetOpt("preset", "ultrafast");
        m_encode.SetOpt("tune", "zerolatency");

        auto re = m_encode.Open();
        if (re < 0) {
            throw std::runtime_error("open encoder failed");
            return;
        }

        m_mux.init_mux(url.c_str(), codecconetxt, "flv");

        m_mux.WriteHead();
    }

    void StreamPusher::set_frommat(int width, int height, int format) {
        from_width = width;
        from_height = height;
        from_format = format;
    }

    void StreamPusher::set_tomat(int width, int height, int format) {
        to_width = width;
        to_height = height;
        to_format = format;
    }

} // namespace FFmpeg
