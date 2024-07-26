//
// Created by lijin on 2023/12/20.
//

#include "FFmpegOutputNode.h"
#include "utils/TimeTicker.h"

#include <utility>

namespace Node {

FFmpegOutputNode::FFmpegOutputNode(std::string name,
                                   std::string open_source,
                                   int         from_width,
                                   int         from_height,
                                   int         from_format,
                                   int         to_width,
                                   int         to_height,
                                   int         to_format,
                                   int         fps,
                                   int         bitrate,
                                   bool        use_hw)
    : m_from_width(from_width),
      m_from_height(from_height),
      m_from_format(from_format),
      m_to_width(to_width),
      m_to_height(to_height),
      m_to_format(to_format),
      m_fps(fps),
      m_bitrate(bitrate),
      m_use_hw(use_hw),
      m_open_source(std::move(open_source)),
      GraphCore::Node(std::move(name), GraphCore::NODE_TYPE::MID_NODE) {}

Data::BaseData::ptr FFmpegOutputNode::handle_data(Data::BaseData::ptr data) {
    TimeTicker();
    av_frame_unref(m_yuv_frame.get());
    auto mat_image = data->MAT_IMAGE;
    if (!m_scaler->scale<cv::Mat, av_frame>(mat_image, m_yuv_frame)) {
        ErrorL << "scale failed";
        return nullptr;
    }
    auto pkt         = alloc_av_packet();
    m_yuv_frame->pts = pts++;
    if (!m_encoder->encode(m_yuv_frame, pkt)) {
        ErrorL << "encode failed";
        return nullptr;
    }
    if (!m_enmux->write_packet(pkt)) {
        ErrorL << "write packet failed";
        m_write_error++;
        if (m_write_error == 100) {
            // 触发重连回调
            error_cb(getName(), GraphCore::StatusCode::NodeError, "输出节点错误，重连中。。。");
            for (int i = 0; i < m_max_reconnect; i++) {
                if (Reconnect()) {
                    WarnL << "重连成功！";
                    m_write_error = 0;
                    break;
                } else {
                    WarnL << "输出节点重连中。。。第" << i << "次";
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
            }
            return nullptr;
        }
    }
    return data;
}

FFmpegOutputNode::~FFmpegOutputNode() {
    Stop();
}
FFmpegOutputNode::ptr FFmpegOutputNode::CreateShared(std::string        name,
                                                     const std::string &open_source,
                                                     int                from_width,
                                                     int                from_height,
                                                     int                from_format,
                                                     int                to_width,
                                                     int                to_height,
                                                     int                to_format,
                                                     int                fps,
                                                     int                bitrate,
                                                     bool               use_hw) {
    return std::make_shared<FFmpegOutputNode>(std::move(name), open_source, from_width, from_height,
                                              from_format, to_width, to_height, to_format, fps,
                                              bitrate, use_hw);
}
bool FFmpegOutputNode::Init() {
    if (m_from_width <= 0 || m_from_height <= 0 || m_to_width <= 0 || m_to_height <= 0) {
        std::cout << "width or height is 0" << std::endl;
        return false;
    }
    if (!m_scaler) {
        m_scaler =
            FFmpeg::Scaler::createShare(m_from_width, m_from_height, (AVPixelFormat)m_from_format,
                                        m_to_width, m_to_height, (AVPixelFormat)m_to_format);
    }
    if (!m_encoder) {
        m_encoder =
            FFmpeg::Encoder::createShared(m_codec_id, m_to_width, m_to_height, m_fps, m_bitrate);
        if (!m_encoder->open(m_use_hw)) {
            std::cout << "encoder open failed" << std::endl;
            return false;
        }
    }
    if (!m_enmux) {
        m_enmux = FFmpeg::Enmuxer::createShared(m_encoder, m_open_source);
        if (!m_enmux->open()) {
            std::cout << "mux open failed" << std::endl;
            return false;
        }
    }
    return true;
}

bool FFmpegOutputNode::Reconnect() {
    m_scaler.reset();
    m_encoder.reset();
    m_enmux.reset();
    return Init();
}

}  // namespace Node