//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_MMP4RECORDNOENCODETASK_H
#define VIDEOPIPELINE_MMP4RECORDNOENCODETASK_H

#include "RecordTask.h"
#include "ffmpeg/core/Encoder.h"
#include "ffmpeg/core/Enmuxer.h"
#include "ffmpeg/core/Scaler.h"

namespace record {

// 为了节约性能，这里接受的数据是推流节点重新编码好的数据，只进行封装写入
// 编码器对象需要从外部传入
class Mp4RecordNoEncodeTask : public RecordTask {
public:
    typedef std::shared_ptr<Mp4RecordNoEncodeTask> ptr;
    std::shared_ptr<FFmpeg::Enmuxer>               m_enmuxer;
    bool                                           m_started = false;

public:
    explicit Mp4RecordNoEncodeTask(RecordConfig config, std::shared_ptr<FFmpeg::Encoder> encoder);
    ~Mp4RecordNoEncodeTask() override;

private:
    void record_handler(Data::BaseData::ptr data) override;
};

}  // namespace record

#endif  // VIDEOPIPELINE_MMP4RECORDNOENCODETASK_H
