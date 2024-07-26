//
// Created by lijin on 2023/12/21.
//

#include "Mp4RecordTask.h"

#include <utility>

namespace record {

void Mp4RecordTask::record_handler(Data::BaseData::ptr data) {
    auto frame     = alloc_av_frame();
    auto mat_image = data->MAT_IMAGE;
    if (!m_scaler->scale<cv::Mat, av_frame>(mat_image, frame)) {
        std::cout << "scale failed" << std::endl;
        return;
    }
    auto pkt   = alloc_av_packet();
    frame->pts = pts++;
    if (!m_encoder->encode(frame, pkt)) {
        std::cout << "encode failed" << std::endl;
        return;
    }
    if (!m_enmuxer->write_packet(pkt)) {
        std::cout << "write packet failed" << std::endl;
        return;
    }
    // 判断任务是否结束
    if (time(nullptr) - m_create_time > m_config.duration) {
        if (record_complete_cb) {
            record_complete_cb("录制完成", 200, m_config);
        }
        m_enmuxer->write_trailer();
        record_status = RecordStatus::COMPLETED;
    }
}

Mp4RecordTask::Mp4RecordTask(RecordConfig config) : RecordTask(std::move(config)) {
    if (m_config.record_type != RecordType::VIDEO_RECORD) {
        throw std::runtime_error("Mp4MuxRecordTask only support video record");
    }
    m_scaler =
        FFmpeg::Scaler::createShare(m_config.src_width, m_config.src_height, AV_PIX_FMT_BGR24,
                                    m_config.dst_width, m_config.dst_height, AV_PIX_FMT_YUV420P);
    m_encoder =
        FFmpeg::Encoder::createShared(m_config.dst_width, m_config.dst_height, 25, 1024 * 1024 * 2);
    if (!m_encoder->open()) {
        throw std::runtime_error("open encoder failed");
    }
    m_enmuxer =
        FFmpeg::Enmuxer::createShared(m_encoder, m_config.save_path + "/" + m_config.file_name);
    if (!m_enmuxer->open()) {
        throw std::runtime_error("open enmux failed");
    }
}

}  // namespace record