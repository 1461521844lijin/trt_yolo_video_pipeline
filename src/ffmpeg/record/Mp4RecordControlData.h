//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_MP4RECORDCONTROLDATA_H
#define VIDEOPIPELINE_MP4RECORDCONTROLDATA_H

#include "ffmpeg/record/RecordTask.h"
#include "graph/core/common/ControlData.h"
namespace Data {

class Mp4RecordControlData : public ControlData {
public:
    using ptr = std::shared_ptr<Mp4RecordControlData>;

    explicit Mp4RecordControlData(record::RecordConfig config)
        : ControlData(ControlType::VIDEO_RECORD), m_record_config(config) {}

    ~Mp4RecordControlData() override = default;

    record::RecordConfig get_record_config() const {
        return m_record_config;
    }

private:
    record::RecordConfig m_record_config;
};

}  // namespace Data
#endif  // VIDEOPIPELINE_MP4RECORDCONTROLDATA_H
