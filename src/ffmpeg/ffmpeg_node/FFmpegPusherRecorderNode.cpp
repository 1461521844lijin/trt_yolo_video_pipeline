//
// Created by lijin on 2023/12/21.
//

#include "FFmpegPusherRecorderNode.h"

#include "graph/object/ImageRecordControlData.h"
#include "graph/object/Mp4RecordControlData.h"
#include "graph/record/JpegRecordTask.h"
#include "graph/record/Mp4RecordNoEncodeTask.h"
#include <utility>
namespace FFmpeg {

FFmpegPusherRecorderNode::~FFmpegPusherRecorderNode() {
    m_record_task_list.clear();
}
FFmpegPusherRecorderNode::FFmpegPusherRecorderNode(std::string name,
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
    : FFmpegOutputNode(std::move(name),
                       std::move(open_source),
                       from_width,
                       from_height,
                       from_format,
                       to_width,
                       to_height,
                       to_format,
                       fps,
                       bitrate,
                       use_hw) {}

Data::BaseData::ptr FFmpegPusherRecorderNode::handle_data(Data::BaseData::ptr data) {
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

    auto new_packet = alloc_av_packet_with(pkt.get());
    data->Insert<AV_PACKET_TYPE>(AV_PACKET, new_packet);
    if (!m_enmux->write_packet(pkt)) {
        std::cout << "write packet failed" << std::endl;
        return nullptr;
    }
    // pkt数据保存后存入record task
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

// Data::BaseData::ptr FFmpegPusherRecorderNode::handle_control_data(Data::ControlData::ptr data) {
//     if (data->get_control_type() == Data::ControlType::IMAGE_RECORD) {
//         auto image_record_data = std::dynamic_pointer_cast<Data::ImageRecordControlData>(data);
//         auto image_record_task =
//             std::make_shared<record::JpegRecordTask>(image_record_data->get_record_config());
//         m_record_task_list.push_back(image_record_task);
//         image_record_task->Start();
//     }
//     if (data->get_control_type() == Data::ControlType::VIDEO_RECORD) {
//         auto mp4_record_data = std::dynamic_pointer_cast<Data::Mp4RecordControlData>(data);
//         auto mp4_record_task = std::make_shared<record::Mp4RecordNoEncodeTask>(
//             mp4_record_data->get_record_config(), m_encoder);
//         m_record_task_list.push_back(mp4_record_task);
//         mp4_record_task->Start();
//     }
//     return data;
// }
}  // namespace FFmpeg