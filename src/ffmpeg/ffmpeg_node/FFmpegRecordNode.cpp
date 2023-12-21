//
// Created by lijin on 2023/12/21.
//

#include "FFmpegRecordNode.h"
#include "graph/object/ImageRecordControlData.h"
#include "graph/object/Mp4RecordControlData.h"
#include "graph/record/JpegRecordTask.h"
#include "graph/record/Mp4RecordTask.h"

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

Data::BaseData::ptr FFmpegRecordNode::handle_control_data(Data::ControlData::ptr data) {
    if (data->get_control_type() == Data::ControlType::IMAGE_RECORD) {
        auto image_record_data = std::dynamic_pointer_cast<Data::ImageRecordControlData>(data);
        auto image_record_task =
            std::make_shared<record::JpegRecordTask>(image_record_data->get_record_config());
        m_record_task_list.push_back(image_record_task);
        image_record_task->Start();
    }
    if (data->get_control_type() == Data::ControlType::VIDEO_RECORD) {
        auto mp4_record_data = std::dynamic_pointer_cast<Data::Mp4RecordControlData>(data);
        auto mp4_record_task =
            std::make_shared<record::Mp4RecordTask>(mp4_record_data->get_record_config());
        m_record_task_list.push_back(mp4_record_task);
        mp4_record_task->Start();
    }
    return data;
}

FFmpegRecordNode::FFmpegRecordNode(const std::string &name) : Node(name) {}

FFmpegRecordNode::~FFmpegRecordNode() {
    m_record_task_list.clear();
}
}  // namespace FFmpeg