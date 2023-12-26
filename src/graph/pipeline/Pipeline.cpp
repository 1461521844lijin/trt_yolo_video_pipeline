//
// Created by lijin on 2023/12/6.
//

#include "Pipeline.h"

namespace pipeline {

void Pipeline::Start() {
    if (!m_initialized) {
        if (!Init()) {
            throw std::runtime_error("Pipeline init failed");
        }
    }
    for (auto &node : m_nodes) {
        node.second->Start();
    }
}

void Pipeline::Stop() {
    for (auto &node : m_nodes) {
        node.second->Stop();
    }
}

Pipeline::~Pipeline() {
    Stop();
}

bool Pipeline::init_from_config(const oatpp::Object<Dto::PipelineDto> &config_dto) {
    m_config_dto = config_dto;
    // todo

    return true;
}

}  // namespace pipeline