//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_YOLOV5DETECTIONINFER_H
#define VIDEOPIPELINE_YOLOV5DETECTIONINFER_H

#include "cuda_kernels/cuda_tools/AffineMatrix.h"
#include "cuda_kernels/cuda_tools/Tensor.h"
#include "cuda_kernels/cuda_tools/monopoly_allocator.hpp"
#include "cuda_kernels/kernels/postprocess/PostProcess.cuh"
#include "cuda_kernels/kernels/preprocess/PreProcess.cuh"
#include "graph/core/common/DetectionBox.h"
#include "graph/core/common/IDetectionInfo.h"
#include "infer/InferPipeline.h"
#include "trt/trt_engine/TRTEngine.h"

namespace infer {

class Yolov5DetectionInfer : public InferPipeline, public IDetectInfo {
public:
    typedef std::shared_ptr<Yolov5DetectionInfer> ptr;

    Yolov5DetectionInfer(std::string infer_name,
                         int         device_id,
                         std::string model_path,
                         std::string label_path,
                         float       score_threshold = 0.5,
                         float       nms_threshold   = 0.5,
                         int         max_batch_size  = 16);

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
    bool                                             m_has_dynamic_dim = false;
    std::vector<int>                                 m_input_shapes;
    std::vector<int>                                 m_output_shapes;
    CUDA::CUStream                                   m_stream = nullptr;
    CUDATools::AffineMatrix                          m_affin_matrix;
    std::shared_ptr<CUDA::Tensor>                    m_affin_matrix_tensor;

    const int MAX_IMAGE_BBOX  = 1024;
    const int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
};

}  // namespace infer

#endif  // VIDEOPIPELINE_YOLOV5DETECTIONINFER_H
