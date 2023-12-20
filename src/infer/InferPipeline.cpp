//
// Created by lijin on 2023/12/20.
//

#include "InferPipeline.h"

#include <utility>

namespace infer {

InferPipeline::InferPipeline(std::string infer_name,
                             std::string model_path,
                             int         device_id,
                             int         max_batch_size)
    : m_infer_name(std::move(infer_name)),
      m_device_id(device_id),
      m_model_path(std::move(model_path)),
      m_max_batch_size(max_batch_size) {
    m_pre_node   = std::make_shared<GraphCore::Node>(m_infer_name + "_pre_process",
                                                   GraphCore::NODE_TYPE::SRC_NODE);
    m_post_node  = std::make_shared<GraphCore::Node>(m_infer_name + "_post_process",
                                                    GraphCore::NODE_TYPE::MID_NODE);
    m_infer_node = std::make_shared<GraphCore::Node>(m_infer_name + "_infer_process",
                                                     GraphCore::NODE_TYPE::DES_NODE);
    m_pre_node->set_get_data_max_num(m_max_batch_size);
    m_post_node->set_get_data_max_num(m_max_batch_size);
    m_infer_node->set_get_data_max_num(m_max_batch_size);
    m_pre_node->set_batch_data_handler_hooker(
        [this](std::vector<Data::BaseData::ptr> &batch_data) -> std::vector<Data::BaseData::ptr> {
            pre_process(batch_data);
            return batch_data;
        });
    m_infer_node->set_batch_data_handler_hooker(
        [this](std::vector<Data::BaseData::ptr> &batch_data) -> std::vector<Data::BaseData::ptr> {
            infer_process(batch_data);
            return batch_data;
        });
    m_post_node->set_batch_data_handler_hooker(
        [this](std::vector<Data::BaseData::ptr> &batch_data) -> std::vector<Data::BaseData::ptr> {
            post_process(batch_data);
            return batch_data;
        });

    auto queue = std::make_shared<GraphCore::ThreadSaveQueue>();
    queue->set_max_size(m_max_batch_size * 2);
    queue->set_buffer_strategy(GraphCore::BufferOverStrategy::BLOCK);
    m_pre_node->add_input(m_infer_name + "_data_input", queue);
    GraphCore::LinkNode(m_pre_node, m_infer_node, m_max_batch_size * 2,
                        GraphCore::BufferOverStrategy::BLOCK);
    GraphCore::LinkNode(m_infer_node, m_post_node, m_max_batch_size * 2,
                        GraphCore::BufferOverStrategy::BLOCK);

    m_pre_node->Start();
    m_post_node->Start();
    m_infer_node->Start();
}
InferPipeline::~InferPipeline() {
    m_pre_node->Stop();
    m_post_node->Stop();
    m_infer_node->Stop();
}

}  // namespace infer