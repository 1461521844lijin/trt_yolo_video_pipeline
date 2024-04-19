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
    // auto node_prt = m_nodes["ffmpeg_input_node"];
    // node_prt->Start();
}

void Pipeline::Stop() {
    for (auto &node : m_nodes) {
        node.second->Stop();
    }
}

Pipeline::~Pipeline() {
    Stop();
}

bool Pipeline::Start_byname(std::string node_name)
{
    auto node_prt = m_nodes[node_name];
    node_prt->Start();
}



}  // namespace pipeline