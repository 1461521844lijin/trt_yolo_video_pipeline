//
// Created by lijin on 2023/12/21.
//

#include "Mp4RecordNoEncodeTask.h"

namespace record {
Mp4RecordNoEncodeTask::Mp4RecordNoEncodeTask(RecordConfig                     config,
                                             std::shared_ptr<FFmpeg::Encoder> encoder)
    : RecordTask(config) {
    if (m_config.record_type != RecordType::VIDEO_RECORD) {
        throw std::runtime_error("record type error");
    }
    m_enmuxer = FFmpeg::Enmuxer::createShared(std::move(encoder),
                                              m_config.save_path + "/" + m_config.file_name);
    if (!m_enmuxer->open()) {
        throw std::runtime_error("open enmux failed");
    }
}

void Mp4RecordNoEncodeTask::record_handler(Data::BaseData::ptr data) {
    auto av_pkt = data->Get<AV_PACKET_TYPE>(AV_PACKET);
    if (!av_pkt) {
        throw std::runtime_error("data is not av packet");
    }
    // 第一次写入的数据帧需要是关键帧
    if (av_pkt->flags & AV_PKT_FLAG_KEY) {
        m_started = true;
    }
    if (m_started) {
        if (!m_enmuxer->write_packet(av_pkt)) {
            std::cout << "write frame failed" << std::endl;
            return;
        }
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

Mp4RecordNoEncodeTask::~Mp4RecordNoEncodeTask() {
    Stop();
    m_enmuxer.reset();
}
}  // namespace record