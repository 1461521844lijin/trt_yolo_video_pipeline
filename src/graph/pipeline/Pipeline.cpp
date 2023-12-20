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
    node->Start();
  }
}

void Pipeline::Stop() {
  for (auto &node : m_nodes) {
    node->Stop();
  }
}

Pipeline::~Pipeline() { Stop(); }

} // namespace pipeline