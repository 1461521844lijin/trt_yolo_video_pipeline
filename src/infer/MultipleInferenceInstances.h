//
// Created by lijin on 2023/12/20.
//

#ifndef VIDEOPIPELINE_MULTIPLEINFERENCEINSTANCES_H
#define VIDEOPIPELINE_MULTIPLEINFERENCEINSTANCES_H

#include "Infer.h"

namespace infer {

template <typename INFER_INSTANCE>
class MultipleInferenceInstances : public Infer {
public:
    using ptr = std::shared_ptr<MultipleInferenceInstances>;

    MultipleInferenceInstances() = delete;

    template <typename... Args>
    explicit MultipleInferenceInstances(const std::string &infer_name,
                                        std::vector<int>   device_list,
                                        Args &&...args) {
        m_infer_name = infer_name;
        for (auto &device_id : device_list) {
            m_device_id_list.push_back(device_id);
            m_infer_list.push_back(std::make_shared<INFER_INSTANCE>(infer_name, device_id,
                                                                    std::forward<Args>(args)...));
        }
    }

    Data::BaseData::ptr commit(const Data::BaseData::ptr &data) override {
        return m_infer_list[get_infer_index()]->commit(data);
    }

    bool init() {
        bool ret = true;
        for (auto &infer_instance : m_infer_list) {
            ret &= infer_instance->init();
        }
        return ret;
    }

private:
    int get_infer_index() {
        m_infer_index = (m_infer_index + 1) % m_infer_list.size();
        return m_infer_index;
    }

private:
    std::vector<std::shared_ptr<INFER_INSTANCE>> m_infer_list;
    std::vector<int>                             m_device_id_list;
    std::atomic_int m_infer_index{0};  // 当前使用的推理设备索引
    std::string     m_infer_name;
};

}  // namespace infer

#endif  // VIDEOPIPELINE_MULTIPLEINFERENCEINSTANCES_H
