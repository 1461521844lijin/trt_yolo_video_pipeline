//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_INFERNODE_H
#define VIDEOPIPELINE_INFERNODE_H

#include "graph/core/node/ProcessNode.h"
#include "infer/Infer.h"

namespace Node {

class InferNode : public GraphCore::Node {
public:
    typedef std::shared_ptr<InferNode> ptr;

    explicit InferNode(std::string name);

    void set_trt_instance(infer::Infer::ptr instance);

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;

private:
    infer::Infer::ptr m_trt_instance;
};

}  // namespace Node

#endif  // VIDEOPIPELINE_INFERNODE_H
