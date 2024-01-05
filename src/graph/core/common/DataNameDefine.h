
//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_DATANAMEDEFINE_H
#define VIDEOPIPELINE_DATANAMEDEFINE_H

#include "ffmpeg/core/SafeAVFormat.h"
#include <opencv2/opencv.hpp>
#include <string>

#include "cuda_kernels/cuda_tools/AffineMatrix.h"
#include "cuda_kernels/cuda_tools/Tensor.h"
#include "graph/core/common/DetectionBox.h"

#define DataDefine(name, type)                                                                     \
    static const std::string name = #name;                                                         \
    typedef type             name##_TYPE;

// 数据表存放定义

// 数据存储格式
DataDefine(MAT_IMAGE, cv::Mat);
DataDefine(AV_PACKET, std::shared_ptr<AVPacket>);
DataDefine(AV_FRAME, std::shared_ptr<AVFrame>);

DataDefine(CUDA_TENSOR, std::shared_ptr<CUDA::Tensor>);
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

#endif                          // VIDEOPIPELINE_DATANAMEDEFINE_H
