//
// Created by lijin on 2024/7/25.
//

#ifndef VIDEOPIPELINE_YOLODETECTIONINFER_H
#define VIDEOPIPELINE_YOLODETECTIONINFER_H

#include "../triton_client/TritonGrpcClient.h"
#include "cuda_kernels/cuda_tools/AffineMatrix.h"
#include "cuda_kernels/cuda_tools/Tensor.h"
#include "cuda_kernels/cuda_tools/monopoly_allocator.hpp"
#include "cuda_kernels/kernels/postprocess/PostProcess.cuh"
#include "cuda_kernels/kernels/preprocess/PreProcess.cuh"
#include "graph/core/common/DetectionBox.h"
#include "graph/core/common/IDetectionInfo.h"
#include "infer/InferInstance.h"

namespace infer {

enum YoloType { V5, X, V3, V7, V8};

class YoloDetectionInfer : public InferInstance, public IDetectInfo {
public:
    typedef std::shared_ptr<YoloDetectionInfer> ptr;

    YoloDetectionInfer(std::string infer_name,
                       int         device_id,
                       std::string triton_url,
                       std::string model_path,
                       std::string label_path,
                       YoloType    type,
                       float       score_threshold = 0.25,
                       float       nms_threshold   = 0.5,
                       int         max_batch_size  = 16);

public:
    /**
     * @brief 初始化推理实例
     * @return 是否初始化成功
     */
    bool init();

    Data::BaseData::ptr commit(const Data::BaseData::ptr &data) override;

private:
    void pre_process(Data::BatchData::ptr &data) override;
    void post_process(Data::BatchData::ptr &batch_data) override;
    void infer_process(Data::BatchData::ptr &batch_data) override;

private:
    void image_to_tensor(Data::BaseData::ptr           &data,
                         std::shared_ptr<CUDA::Tensor> &tensor,
                         int                            ibatch);

private:
    triton_client::TritonGrpcClient::ptr             m_triton_client;
    std::shared_ptr<MonopolyAllocator<CUDA::Tensor>> m_tensor_allocator = std::make_shared<MonopolyAllocator<CUDA::Tensor>>(20);
    std::vector<int64_t>                             m_input_shapes;
    std::vector<int64_t>                             m_output_shapes;

    YoloType         m_type;
    CUDA::Norm       m_normalize;

    const int MAX_IMAGE_BBOX  = 1024;
    const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class, keepflag

    std::string m_triton_url;
};

}  // namespace infer

#endif  // VIDEOPIPELINE_YOLODETECTIONINFER_H
