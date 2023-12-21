//
// Created by lijin on 2023/12/21.
//

#include "JpegRecordTask.h"

#include <utility>

namespace record {
JpegRecordTask::JpegRecordTask(RecordConfig config) : RecordTask(std::move(config)) {}

void JpegRecordTask::record_handler(Data::BaseData::ptr data) {
    try {
        auto    mat_image = data->Get<MAT_IMAGE_TYPE>(MAT_IMAGE);
        cv::Mat image;
        cv::resize(mat_image, image, cv::Size(m_config.dst_width, m_config.dst_height));
        cv::imwrite(m_config.save_path + "/" + m_config.file_name, image);
        if (record_complete_cb) {
            record_complete_cb("图片保存完成", 200, m_config);
        }
    } catch (std::exception &e) {
        if (record_complete_cb) {
            record_complete_cb(e.what(), 500, m_config);
        }
    }
    record_status = RecordStatus::COMPLETED;
}

JpegRecordTask::~JpegRecordTask() {
    Stop();
}
}  // namespace record