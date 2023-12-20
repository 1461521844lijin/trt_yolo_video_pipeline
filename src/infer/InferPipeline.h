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

class InferPipeline : public Infer {
public:
    using ptr       = std::shared_ptr<InferPipeline>;
    InferPipeline() = delete;

    explicit InferPipeline(std::string infer_name,
                           std::string model_path,
                           int         device_id      = 0,
                           int         max_batch_size = 16);
    virtual ~InferPipeline();

protected:
    virtual void pre_process(std::vector<Data::BaseData::ptr> &batch_data) {}
    virtual void post_process(std::vector<Data::BaseData::ptr> &batch_data) {}
    virtual void infer_process(std::vector<Data::BaseData::ptr> &batch_data) {}

private:
    GraphCore::Node::ptr m_pre_node;
    GraphCore::Node::ptr m_post_node;
    GraphCore::Node::ptr m_infer_node;

private:
    std::string m_infer_name;
    std::string m_model_path;
    int         m_device_id      = 0;
    int         m_max_batch_size = 16;
};

}  // namespace infer

#endif  // VIDEOPIPELINE_INFERIMP_H
