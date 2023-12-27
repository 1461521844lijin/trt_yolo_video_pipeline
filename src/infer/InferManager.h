//
// Created by lijin on 2023/12/25.
//

#ifndef VIDEOPIPELINE_INFERMANAGER_H
#define VIDEOPIPELINE_INFERMANAGER_H

#include "infer/Infer.h"
#include <mutex>
#include <unordered_map>

namespace infer {

class InferManager {
public:
    using ptr = std::shared_ptr<InferManager>;

    InferManager() = default;

public:
    void add_infer_instance(const std::string &inference_name, const Infer::ptr &infer_instance);

    Infer::ptr get_infer_instance(const std::string &inference_name);

    void remove_infer_instance(const std::string &inference_name);

    void clear();

private:
    std::mutex                                  m_mutex;
    std::unordered_map<std::string, Infer::ptr> m_infer_instances;
};

}  // namespace infer

#endif  // VIDEOPIPELINE_INFERMANAGER_H
