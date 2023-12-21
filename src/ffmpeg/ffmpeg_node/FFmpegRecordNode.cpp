//
// Created by lijin on 2023/12/21.
//

#include "FFmpegRecordNode.h"
#include "ffmpeg/record/ImageRecordControlData.h"
#include "ffmpeg/record/JpegRecordTask.h"
#include "ffmpeg/record/Mp4RecordControlData.h"
#include "ffmpeg/record/Mp4RecordTask.h"

namespace FFmpeg {
Data::BaseData::ptr FFmpegRecordNode::handle_data(Data::BaseData::ptr data) {
    for (auto task = m_record_task_list.begin(); task != m_record_task_list.end();) {
        if ((*task)->get_record_status() == record::RecordStatus::COMPLETED) {
            task = m_record_task_list.erase(task);
        } else {
            (*task)->push(data);
            task++;
        }
    }
    return data;
}

FFmpegRecordNode::FFmpegRecordNode(const std::string &name)
    : Node(name, GraphCore::NODE_TYPE::DES_NODE) {
    auto recode_handle_cb = [this](Data::BaseData::ptr data) {
        auto record_data = std::dynamic_pointer_cast<Data::ControlData>(data);
        if (record_data->get_control_type() == Data::ControlType::IMAGE_RECORD) {
            auto image_record_data =
                std::dynamic_pointer_cast<Data::ImageRecordControlData>(record_data);
            auto image_record_task =
                std::make_shared<record::JpegRecordTask>(image_record_data->get_record_config());
            m_record_task_list.push_back(image_record_task);
            image_record_task->Start();
        }
        if (record_data->get_control_type() == Data::ControlType::VIDEO_RECORD) {
            auto mp4_record_data =
                std::dynamic_pointer_cast<Data::Mp4RecordControlData>(record_data);
            auto mp4_record_task =
                std::make_shared<record::Mp4RecordTask>(mp4_record_data->get_record_config());
            m_record_task_list.push_back(mp4_record_task);
            mp4_record_task->Start();
        }
    };
    set_extra_input_callback(recode_handle_cb);
}

FFmpegRecordNode::~FFmpegRecordNode() {
    m_record_task_list.clear();
}
}  // namespace FFmpeg