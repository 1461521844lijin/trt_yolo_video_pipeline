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

#include "DataNameDefine.h"
#include "cuda_kernels/cuda_tools/monopoly_allocator.hpp"

namespace Data {
// 数据表存放定义
#define DataDefine(name, type) type name;

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
    // 数据存储格式
    DataDefine(MAT_IMAGE, cv::Mat);
    DataDefine(AV_PACKET, std::shared_ptr<AVPacket>);
    DataDefine(AV_FRAME, std::shared_ptr<AVFrame>);

    DataDefine(CUDA_AFFINMATRIX_TENSOR, std::shared_ptr<CUDA::Tensor>);
    DataDefine(CUDA_AFFINMATRIX, CUDATools::AffineMatrix);

    // 数据来源

    // 数据处理结果
    DataDefine(DETECTBOX_FUTURE, std::shared_future<DetectBoxArray>);
    DataDefine(DETECTBOX_PROMISE, std::shared_ptr<std::promise<DetectBoxArray>>);

    // 数据信息
    DataDefine(FRAME_INDEX, int);   // 帧序号，int
    DataDefine(FRAME_WIDTH, int);   // 帧宽度，int
    DataDefine(FRAME_HEIGHT, int);  // 帧高度，int
};


class BatchData : public BaseData {
public:
    using ptr = std::shared_ptr<BatchData>;

    BatchData() = delete;

    explicit BatchData(DataType data_type) : BaseData(data_type) {}


    std::vector<BaseData::ptr> batch_data;

    DataDefine(BATCH_INPUT_TENSOR, MonopolyAllocator<CUDA::Tensor>::MonopolyDataPointer);
    DataDefine(BATCH_OUTPUT_TENSOR, MonopolyAllocator<CUDA::Tensor>::MonopolyDataPointer);
};


}  // namespace Data

#endif  // TRT_YOLOV8_SERVER_BASEDATA_H
