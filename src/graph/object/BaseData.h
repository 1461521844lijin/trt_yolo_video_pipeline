//
// Created by lijin on 2023/8/3.
//

#ifndef TRT_YOLOV8_SERVER_BASEDATA_H
#define TRT_YOLOV8_SERVER_BASEDATA_H

#include <chrono>
#include <future>
#include <memory>

namespace GraphCore {

enum DataType {
    DATATYPE_UNKNOWN = 0,
    FRAME,    // 帧数据
    CONTROL,  // 控制数据
    CONFIG,   // 配置数据
    DATATYPE_MAX
};

// 用于存储数据的基类
class BaseData {
public:
    using ptr = std::shared_ptr<BaseData>;

    BaseData() = delete;

    explicit BaseData(DataType data_type) {
        this->data_type = data_type;
        create_time     = std::chrono::system_clock::now();
    }

    virtual ~BaseData() = default;

    DataType get_data_type() const {
        return data_type;
    }

    DataType                              data_type;    // 数据类型
    std::chrono::system_clock::time_point create_time;  // 数据创建时间
    std::string                           data_name;    // 数据名称/数据来源
};

}  // namespace Data

#endif  // TRT_YOLOV8_SERVER_BASEDATA_H
