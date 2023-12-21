//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_IMAGERECORDCONTROLDATA_H
#define VIDEOPIPELINE_IMAGERECORDCONTROLDATA_H

#include "ffmpeg/record/RecordTask.h"
#include "graph/core/common/ControlData.h"

namespace Data {

class ImageRecordControlData : public ControlData {
public:
    using ptr = std::shared_ptr<ImageRecordControlData>;

    explicit ImageRecordControlData(record::RecordConfig config)
        : ControlData(ControlType::IMAGE_RECORD), m_record_config(config) {}

    ~ImageRecordControlData() override = default;

    record::RecordConfig get_record_config() const {
        return m_record_config;
    }

private:
    record::RecordConfig m_record_config;
};

}  // namespace Data

#endif  // VIDEOPIPELINE_IMAGERECORDCONTROLDATA_H
