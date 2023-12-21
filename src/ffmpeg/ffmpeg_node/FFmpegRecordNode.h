//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_RECORDNODE_H
#define VIDEOPIPELINE_RECORDNODE_H

#include "ffmpeg/record/RecordTask.h"
#include "graph/core/node/ProcessNode.h"
namespace Node {

class FFmpegRecordNode : public GraphCore::Node {
public:
    typedef std::shared_ptr<FFmpegRecordNode> ptr;
    std::list<record::RecordTask::ptr>        m_record_task_list;
    explicit FFmpegRecordNode(const std::string &name);
    ~FFmpegRecordNode();

private:
protected:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
};

}  // namespace Node

#endif  // VIDEOPIPELINE_RECORDNODE_H
