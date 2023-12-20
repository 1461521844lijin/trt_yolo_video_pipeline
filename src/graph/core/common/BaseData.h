//
// Created by lijin on 2023/8/3.
//

#ifndef TRT_YOLOV8_SERVER_BASEDATA_H
#define TRT_YOLOV8_SERVER_BASEDATA_H

#include <chrono>
#include <future>
#include <memory>
#include <vector>

#include <any>
#include <unordered_map>

namespace Data {

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

public:
    // 通过name插入任意类型的数据
    template <typename T>
    void Insert(const std::string &name, const T &value) {
        data_map[name] = value;
    }

    // 通过name获取任意类型的数据
    template <typename T>
    T Get(const std::string &name) {
        try {
            std::any_cast<T>(data_map[name]);
        } catch (const std::bad_any_cast &e) {
            throw std::runtime_error("类型转换错误，请检查数据类型: " + std::string(e.what()));
        }
    }

private:
    std::unordered_map<std::string, std::any> data_map;
};

}  // namespace GraphCore

#endif  // TRT_YOLOV8_SERVER_BASEDATA_H
