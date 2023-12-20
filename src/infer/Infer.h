//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_INFER_H
#define VIDEOPIPELINE_INFER_H

#include "graph/core/common/BaseData.h"
#include <vector>

namespace infer {

class Infer {
public:
    using ptr = std::shared_ptr<Infer>;

public:
    virtual Data::BaseData::ptr commit(const Data::BaseData::ptr &data) = 0;

    std::vector<Data::BaseData::ptr> commits(const std::vector<Data::BaseData::ptr> &batch_data) {
        std::vector<Data::BaseData::ptr> results;
        results.reserve(batch_data.size());
        for (auto &data : batch_data) {
            results.push_back(commit(data));
        }
        return results;
    }
};

}  // namespace infer

#endif  // VIDEOPIPELINE_INFER_H
