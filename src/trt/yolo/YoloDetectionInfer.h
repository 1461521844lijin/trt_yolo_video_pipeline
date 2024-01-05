//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_YOLODETECTIONINFER_H
#define VIDEOPIPELINE_YOLODETECTIONINFER_H

#include "cuda_kernels/cuda_tools/AffineMatrix.h"
#include "cuda_kernels/cuda_tools/Tensor.h"
#include "cuda_kernels/cuda_tools/monopoly_allocator.hpp"
#include "cuda_kernels/kernels/postprocess/PostProcess.cuh"
#include "cuda_kernels/kernels/preprocess/PreProcess.cuh"
#include "graph/core/common/DetectionBox.h"
#include "graph/core/common/IDetectionInfo.h"
#include "infer/InferInstance.h"
#include "trt/trt_engine/TRTEngine.h"

namespace infer {

enum YoloType { V5, X, V3, V7, V8, V8Seg };

class YoloDetectionInfer : public InferInstance, public IDetectInfo {
public:
    typedef std::shared_ptr<YoloDetectionInfer> ptr;

    YoloDetectionInfer(std::string infer_name,
                       int         device_id,
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
    void pre_process(std::vector<Data::BaseData::ptr> &batch_data) override;
    void post_process(std::vector<Data::BaseData::ptr> &batch_data) override;
    void infer_process(std::vector<Data::BaseData::ptr> &batch_data) override;

private:
    void image_to_tensor(const cv::Mat &image, std::shared_ptr<CUDA::Tensor> &tensor, int ibatch);

private:
    TRT::TRTEngine::ptr                              m_trt_engine;
    std::shared_ptr<MonopolyAllocator<CUDA::Tensor>> m_tensor_allocator;  // 输入显存分配器
    std::shared_ptr<CUDA::Tensor>                    m_input_tensor;
    std::shared_ptr<CUDA::Tensor>                    m_output_tensor;
    std::shared_ptr<CUDA::Tensor>                    m_segment_tensor;
    std::shared_ptr<CUDA::Tensor>                    m_segment_tensor_cache;
    std::vector<int>                                 m_input_shapes;
    std::vector<int>                                 m_output_shapes;
    CUDA::CUStream                                   m_stream = nullptr;
    CUDATools::AffineMatrix                          m_affin_matrix;
    std::shared_ptr<CUDA::Tensor>                    m_affin_matrix_tensor;
    YoloType                                         m_type;
    bool                                             m_dynamic     = false;
    bool                                             m_has_segmegt = false;
    std::vector<int>                                 m_segment_shapes;
    CUDA::Norm                                       m_normalize;

    const int MAX_IMAGE_BBOX  = 1024;
    const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class, keepflag
};

}  // namespace infer

#endif  // VIDEOPIPELINE_YOLODETECTIONINFER_H
