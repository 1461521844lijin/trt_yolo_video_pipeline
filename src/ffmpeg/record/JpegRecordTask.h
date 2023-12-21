//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_JPEGRECORDTASK_H
#define VIDEOPIPELINE_JPEGRECORDTASK_H

#include "RecordTask.h"
#include "ffmpeg/ImageSaver.h"

namespace record {

class JpegRecordTask : public RecordTask {
public:
    typedef std::shared_ptr<JpegRecordTask> ptr;

public:
    explicit JpegRecordTask(RecordConfig config);
    ~JpegRecordTask() override;

private:
    void record_handler(Data::BaseData::ptr data) override;
};

}  // namespace record

#endif  // VIDEOPIPELINE_JPEGRECORDTASK_H
