//
// Created by lijin on 2023/12/20.
//

#include "FFmpegReadNode.h"
#include "graph/core/common/StatusCode.h"
#include <utility>

namespace FFmpeg {
FFmpegReadNode::FFmpegReadNode(const std::string &name,
                               std::string        open_source,
                               bool               use_hw,
                               bool               cycle)
    : GraphCore::Node(name, GraphCore::NODE_TYPE::SRC_NODE),
      m_open_source(std::move(open_source)),
      m_use_hw(use_hw),
      m_cycle(cycle) {
    auto init_cb = [this](const std::string &name, int code, const std::string &msg) {
        if (!m_demux) {
            m_demux = Demuxer::createShare();
        }
        if (!(m_demux->open(m_open_source))) {
            std::cout << "open url " << m_open_source << "failed" << std::endl;
            return -1;
        }
        if (!m_scaler) {
            m_scaler =
                Scaler::createShare(m_demux->get_video_codec_parameters()->width,
                                    m_demux->get_video_codec_parameters()->height,
                                    (AVPixelFormat)m_demux->get_video_codec_parameters()->format,
                                    m_demux->get_video_codec_parameters()->width,
                                    m_demux->get_video_codec_parameters()->height,
                                    AV_PIX_FMT_BGR24);
        }
        if (!m_decoder) {
            m_decoder = Decoder::createShare(m_demux);
        }
        if (!(m_decoder->open(m_use_hw))) {
            return -1;
        }
        m_width  = m_demux->get_video_codec_parameters()->width;
        m_height = m_demux->get_video_codec_parameters()->height;
        m_fps    = m_demux->get_video_stream()->avg_frame_rate.num /
                m_demux->get_video_stream()->avg_frame_rate.den;
        m_bitrate = m_demux->get_video_codec_parameters()->bit_rate;
        return 0;
    };

    set_before_start_cb(init_cb);
}

void FFmpegReadNode::worker() {
    int frame_index = 0;
    while (m_run) {
        auto pkt = alloc_av_packet();
        int  re  = m_demux->read_packet(pkt);
        if (re == EXIT_SUCCESS) {
            if (pkt->stream_index != m_demux->get_video_stream_index()) {
                continue;  // 忽略非视频帧
            }
            m_decoder->send(pkt);
            auto frame = alloc_av_frame();
            if (!m_decoder->receive(frame)) {
                continue;  // 编码器前几帧的缓存可能接收不到
            }
            cv::Mat image(frame->height, frame->width, CV_8UC3);
            if (!m_scaler->scale<av_frame, cv::Mat>(frame, image)) {
                std::cout << "scale failed" << std::endl;
                continue;
            }
            auto data = std::make_shared<Data::BaseData>(Data::DataType::FRAME);
            data->Insert<MAT_IMAGE_TYPE>(MAT_IMAGE, image);
            data->Insert<FRAME_INDEX_TYPE>(FRAME_INDEX, frame_index++);
            data->Insert<FRAME_WIDTH_TYPE>(FRAME_WIDTH, frame->width);
            data->Insert<FRAME_HEIGHT_TYPE>(FRAME_HEIGHT, frame->height);
            send_output_data(data);
        } else if (re == AVERROR_EOF) {
            std::cout << "read eof" << std::endl;
            if (m_cycle) {
                m_demux->seek(0);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            break;
        } else {
            std::cout << "read error" << std::endl;
            error_cb(getName(), GraphCore::StatusCode::NodeError, "读取节点错误，线程退出");
            break;
        }
    }
}

FFmpegReadNode::ptr FFmpegReadNode::CreateShared(const std::string &name,
                                                 std::string        open_source,
                                                 bool               use_hw,
                                                 bool               cycle) {
    return std::make_shared<FFmpegReadNode>(name, std::move(open_source), use_hw, cycle);
}

std::tuple<int, int, int, int64_t> FFmpegReadNode::get_video_info() const {
    return std::make_tuple(m_width, m_height, m_fps, m_bitrate);
}

FFmpegReadNode::~FFmpegReadNode() {
    Stop();
    m_demux.reset();
    m_decoder.reset();
    m_scaler.reset();
}

}  // namespace FFmpeg