//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_MP4RECORDTASK_H
#define VIDEOPIPELINE_MP4RECORDTASK_H

#include "RecordTask.h"
#include "ffmpeg/core/Encoder.h"
#include "ffmpeg/core/Enmuxer.h"
#include "ffmpeg/core/Scaler.h"

namespace record {

class Mp4RecordTask : public RecordTask {
public:
    typedef std::shared_ptr<Mp4RecordTask> ptr;
    std::shared_ptr<FFmpeg::Enmuxer>       m_enmuxer;
    std::shared_ptr<FFmpeg::Encoder>       m_encoder;
    std::shared_ptr<FFmpeg::Scaler>        m_scaler;
    int                                    pts = 0;

public:
    explicit Mp4RecordTask(RecordConfig config);
    ~Mp4RecordTask() override;

private:
    void record_handler(Data::BaseData::ptr data) override;
};

}  // namespace record

#endif  // VIDEOPIPELINE_MP4RECORDTASK_H
