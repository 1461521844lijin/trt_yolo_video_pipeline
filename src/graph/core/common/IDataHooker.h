//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_IDATAHOOKER_H
#define VIDEOPIPELINE_IDATAHOOKER_H

#include "BaseData.h"
#include <functional>
#include <utility>

namespace GraphCore {

typedef std::function<Data::BaseData::ptr(Data::BaseData::ptr)> DataHookerFunc;
typedef std::function<std::vector<Data::BaseData::ptr>(std::vector<Data::BaseData::ptr> &)>
    BatchDataHookerFunc;

class IDataHooker {
protected:
    // 数据处理Hooker，用于替换默认的数据处理函数
    DataHookerFunc data_handler_hooker = nullptr;

    // batch数据处理完毕Hooker
    BatchDataHookerFunc batch_data_handler_hooker = nullptr;

public:
    void set_data_handler_hooker(DataHookerFunc data_hooker) {
        data_handler_hooker = std::move(data_hooker);
    }

    void set_batch_data_handler_hooker(BatchDataHookerFunc batch_data_hooker) {
        batch_data_handler_hooker = std::move(batch_data_hooker);
    }
};

}  // namespace GraphCore

#endif  // VIDEOPIPELINE_IDATAHOOKER_H
