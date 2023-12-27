//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_STREAMPUSHERANDRECORDER_H
#define VIDEOPIPELINE_STREAMPUSHERANDRECORDER_H

#include "FFmpegOutputNode.h"
#include "ffmpeg/core/Encoder.h"
#include "ffmpeg/core/Enmuxer.h"
#include "ffmpeg/core/Scaler.h"
#include "ffmpeg/record/RecordTask.h"
#include "graph/core/node/ProcessNode.h"

namespace Node {

/**
 * @brief 推流和视频录制封装在一起，录制mp4的数据复用推流的数据，节约性能
 *        缺点是录制视频的尺寸和推流的尺寸必须一致
 */
class FFmpegPusherRecorderNode : public FFmpegOutputNode {
public:
    typedef std::shared_ptr<FFmpegPusherRecorderNode> ptr;

    explicit FFmpegPusherRecorderNode(std::string name,
                                      std::string open_source,
                                      int         from_width,
                                      int         from_height,
                                      int         from_format,
                                      int         to_width,
                                      int         to_height,
                                      int         to_format,
                                      int         fps     = 25,
                                      int         bitrate = 1024 * 1024 * 2,
                                      bool        use_hw  = false);

    ~FFmpegPusherRecorderNode();

private:
    std::list<record::RecordTask::ptr> m_record_task_list;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
};

}  // namespace Node

#endif  // VIDEOPIPELINE_STREAMPUSHERANDRECORDER_H
