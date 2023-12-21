//
// Created by lijin on 2023/12/20.
//

#include "FFmpegOutputNode.h"

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
      GraphCore::Node(std::move(name), GraphCore::NODE_TYPE::MID_NODE) {
    auto init_cb = [this](const std::string &name, int code, const std::string &msg) {
        if (m_from_width <= 0 || m_from_height <= 0 || m_to_width <= 0 || m_to_height <= 0) {
            std::cout << "width or height is 0" << std::endl;
            return -1;
        }
        if (!m_scaler) {
            m_scaler = FFmpeg::Scaler::createShare(m_from_width, m_from_height,
                                                   (AVPixelFormat)m_from_format, m_to_width,
                                                   m_to_height, (AVPixelFormat)m_to_format);
        }
        if (!m_encoder) {
            m_encoder = FFmpeg::Encoder::createShared(m_codec_id, m_to_width, m_to_height, m_fps,
                                                      m_bitrate);
            if (!m_encoder->open(m_use_hw)) {
                std::cout << "encoder open failed" << std::endl;
                return -1;
            }
        }
        if (!m_enmux) {
            m_enmux = FFmpeg::Enmuxer::createShared(m_encoder, m_open_source);
            if (!m_enmux->open()) {
                std::cout << "mux open failed" << std::endl;
                return -1;
            }
        }
        return 0;
    };
    set_before_start_cb(init_cb);
}

Data::BaseData::ptr FFmpegOutputNode::handle_data(Data::BaseData::ptr data) {
    av_frame_unref(m_yuv_frame.get());
    auto mat_image = data->Get<MAT_IMAGE_TYPE>(MAT_IMAGE);
    if (!m_scaler->scale<cv::Mat, av_frame>(mat_image, m_yuv_frame)) {
        std::cout << "scale failed" << std::endl;
        return nullptr;
    }
    auto pkt         = alloc_av_packet();
    m_yuv_frame->pts = pts++;
    if (!m_encoder->encode(m_yuv_frame, pkt)) {
        std::cout << "encode failed" << std::endl;
        return nullptr;
    }
    if (!m_enmux->write_packet(pkt)) {
        std::cout << "write packet failed" << std::endl;
        return nullptr;
    }
    return data;
}

FFmpegOutputNode::~FFmpegOutputNode() {
    Stop();
    m_encoder.reset();
    m_scaler.reset();
    m_enmux.reset();
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
}  // namespace Node