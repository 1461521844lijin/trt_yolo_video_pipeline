//
// Created by lijin on 2023/12/20.
//

#ifndef VIDEOPIPELINE_INFERIMP_H
#define VIDEOPIPELINE_INFERIMP_H

#include "Infer.h"
#include "graph/core/node/ProcessNode.h"
#include <future>
#include <memory>
#include <string>

namespace infer {

class InferInstance : public Infer {
public:
    using ptr       = std::shared_ptr<InferInstance>;
    InferInstance() = delete;

    explicit InferInstance(std::string infer_name,
                           std::string model_path,
                           int         device_id      = 0,
                           int         max_batch_size = 16);
    virtual ~InferInstance();

public:
protected:
    virtual void pre_process(Data::BatchData::ptr &batch_data) {}
    virtual void post_process(Data::BatchData::ptr &batch_data) {}
    virtual void infer_process(Data::BatchData::ptr &batch_data) {}

protected:
    GraphCore::Node::ptr m_infer_node;

protected:
    std::string m_infer_name;
    std::string m_model_path;
    int         m_device_id      = 0;
    int         m_max_batch_size = 16;
};

}  // namespace infer

#endif  // VIDEOPIPELINE_INFERIMP_H
