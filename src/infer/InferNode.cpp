//
// Created by lijin on 2023/12/21.
//

#include "InferNode.h"

namespace Node {

InferNode::InferNode(std::string name) : Node(std::move(name), GraphCore::NODE_TYPE::MID_NODE) {}
void InferNode::set_trt_instance(infer::Infer::ptr instance) {
    m_trt_instance = instance;
}
Data::BaseData::ptr InferNode::handle_data(Data::BaseData::ptr data) {
    return m_trt_instance->commit(data);
}

}  // namespace Node