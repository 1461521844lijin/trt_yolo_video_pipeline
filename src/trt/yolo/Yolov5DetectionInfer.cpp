//
// Created by lijin on 2023/12/21.
//

#include "Yolov5DetectionInfer.h"
#include "cuda_kernels/cuda_tools/cuda_tools.h"
#include "graph/core/common/DetectionBox.h"

#include <utility>

namespace infer {

Yolov5DetectionInfer::Yolov5DetectionInfer(std::string infer_name,
                                           int         device_id,
                                           std::string model_path,
                                           std::string label_path,
                                           float       score_threshold,
                                           float       nms_threshold,
                                           int         max_batch_size)
    : InferPipeline(std::move(infer_name), std::move(model_path), device_id, max_batch_size) {
    load_class_names(label_path);
    set_confidence_threshold(score_threshold);
    set_nms_threshold(nms_threshold);
    m_trt_engine       = TRT::TRTEngine::CreateShared(m_model_path, m_device_id);
    m_has_dynamic_dim  = m_trt_engine->has_dynamic_dim();
    m_input_shapes     = m_trt_engine->run_dims(0);
    m_output_shapes    = m_trt_engine->run_dims(1);
    m_tensor_allocator = std::make_shared<MonopolyAllocator<CUDA::Tensor>>(m_max_batch_size * 2);
}
void Yolov5DetectionInfer::pre_process(std::vector<Data::BaseData::ptr> &batch_data) {
    int batch_size = batch_data.size();
    m_input_tensor->resize(batch_size, 3, m_input_shapes[2], m_input_shapes[3]);
    m_output_tensor->resize(batch_size, m_output_shapes[1], m_output_shapes[2], m_output_shapes[3]);
    for (int i = 0; i < batch_size; ++i) {
        auto &image = batch_data[i]->Get<MAT_IMAGE_TYPE>(MAT_IMAGE);
        image_to_tensor(image, m_input_tensor, i);
    }
}

void Yolov5DetectionInfer::infer_process(std::vector<Data::BaseData::ptr> &batch_data) {
    std::vector<void *> bindings = {m_input_tensor->gpu(), m_output_tensor->gpu()};
    if (!m_trt_engine->forward(bindings, nullptr, nullptr)) {
        std::cout << "forward failed" << std::endl;
        batch_data.clear();
        return;
    }
}

void Yolov5DetectionInfer::post_process(std::vector<Data::BaseData::ptr> &batch_data) {
    int          batch_size = batch_data.size();
    CUDA::Tensor output_array_device(CUDA::DataType::Float);
    output_array_device.resize(batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();
    for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
        float *image_based_output = m_output_tensor->gpu<float>(ibatch);
        float *output_array_ptr   = output_array_device.gpu<float>(ibatch);
        auto   affine_matrix      = m_affin_matrix_tensor->gpu<float>();
        checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), m_stream));
        CUDA::yolov5_decode_kernel_invoker(image_based_output, m_output_tensor->size(1),
                                           m_output_tensor->size(2), m_confidence_threshold,
                                           affine_matrix, output_array_ptr, MAX_IMAGE_BBOX,
                                           m_stream);
        CUDA::nms_kernel_invoker(output_array_ptr, m_nms_threshold, MAX_IMAGE_BBOX, m_stream);
    }

    output_array_device.to_cpu();
    for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
        float         *parray = output_array_device.cpu<float>(ibatch);
        int            count  = std::min(MAX_IMAGE_BBOX, (int)*parray);
        DetectBoxArray boxArray(count);
        for (int i = 0; i < count; ++i) {
            float *pbox     = parray + 1 + i * NUM_BOX_ELEMENT;
            int    label    = pbox[5];
            int    keepflag = pbox[6];
            if (keepflag == 1) {
                // x y w h conf label
                boxArray.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
            }
        }
        batch_data[ibatch]->Get<DETECTBOX_PROMISE_TYPE>(DETECTBOX_PROMISE)->set_value(boxArray);
    }
}

void Yolov5DetectionInfer::image_to_tensor(const cv::Mat                 &image,
                                           std::shared_ptr<CUDA::Tensor> &tensor,
                                           int                            ibatch) {
    CUDA::Norm normalize;

    normalize = CUDA::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDA::ChannelType::Invert);
    cv::Size input_size(tensor->size(3), tensor->size(2));

    m_affin_matrix.compute(image.size(), input_size);
    size_t   size_image           = image.cols * image.rows * 3;
    size_t   size_matrix          = iLogger::upbound(sizeof(m_affin_matrix.d2i), 32);
    auto     workspace            = tensor->get_workspace();
    uint8_t *gpu_workspace        = (uint8_t *)workspace->gpu(size_matrix + size_image);
    float   *affine_matrix_device = (float *)gpu_workspace;
    uint8_t *image_device         = size_matrix + gpu_workspace;

    uint8_t *cpu_workspace      = (uint8_t *)workspace->cpu(size_matrix + size_image);
    float   *affine_matrix_host = (float *)cpu_workspace;
    uint8_t *image_host         = size_matrix + cpu_workspace;
    auto     stream             = tensor->get_stream();

    memcpy(image_host, image.data, size_image);
    memcpy(affine_matrix_host, m_affin_matrix.d2i, sizeof(m_affin_matrix.d2i));
    checkCudaRuntime(
        cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
    checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host,
                                     sizeof(m_affin_matrix.d2i), cudaMemcpyHostToDevice, stream));

    checkCudaRuntime(cudaMemcpyAsync(m_affin_matrix_tensor->gpu(), affine_matrix_device,
                                     sizeof(m_affin_matrix.d2i), cudaMemcpyDeviceToDevice));

    CUDA::warpAffineBilinearAndNormalizePlaneInvoker(image_device, image.cols * 3, image.cols,
                                                     image.rows, tensor->gpu<float>(ibatch),
                                                     input_size.width, input_size.height,
                                                     affine_matrix_device, 114, normalize, stream);
    tensor->synchronize();
}
}  // namespace infer