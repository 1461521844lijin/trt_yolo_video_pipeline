//
// Created by lijin on 2023/12/25.
//

#include "InferManager.h"

namespace infer {
void InferManager::add_infer_instance(const std::string &inference_name,
                                      const Infer::ptr  &infer_instance) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_infer_instances.find(inference_name) == m_infer_instances.end()) {
        m_infer_instances[inference_name] = infer_instance;
    } else {
        std::cerr << "Infer instance " << inference_name << " already exists";
    }
}

Infer::ptr InferManager::get_infer_instance(const std::string &inference_name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_infer_instances.find(inference_name) != m_infer_instances.end()) {
        return m_infer_instances[inference_name];
    } else {
        std::cerr << "Infer instance " << inference_name << " not exists";
        return nullptr;
    }
}

void InferManager::remove_infer_instance(const std::string &inference_name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_infer_instances.find(inference_name) != m_infer_instances.end()) {
        m_infer_instances.erase(inference_name);
    } else {
        std::cerr << "Infer instance " << inference_name << " not exists";
    }
}

void InferManager::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_infer_instances.clear();
}
}  // namespace infer