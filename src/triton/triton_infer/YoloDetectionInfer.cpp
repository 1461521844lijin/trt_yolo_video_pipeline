//
// Created by lijin on 2023/12/21.
//

#include "YoloDetectionInfer.h"
#include "cuda_kernels/cuda_tools/cuda_tools.h"
#include "graph/core/common/DetectionBox.h"
#include "utils/TimeTicker.h"
#include "utils/logger.h"

#include <utility>

namespace infer {

static void affine_project(float *matrix, float x, float y, float *ox, float *oy) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

YoloDetectionInfer::YoloDetectionInfer(std::string infer_name,
                                       int         device_id,
                                       std::string triton_url,
                                       std::string model_path,
                                       std::string label_path,
                                       YoloType    type,
                                       float       score_threshold,
                                       float       nms_threshold,
                                       int         max_batch_size)
    : InferInstance(std::move(infer_name), std::move(model_path), device_id, max_batch_size) {
    load_class_names(label_path);
    set_confidence_threshold(score_threshold);
    set_nms_threshold(nms_threshold);
    m_type       = type;
    m_triton_url = std::move(triton_url);
}

bool YoloDetectionInfer::init() {
    m_triton_client = std::make_shared<triton_client::TritonGrpcClient>();
    if (!m_triton_client->init_client(m_triton_url, false)) {
        return false;
    }
    if (!m_triton_client->init_model_infer(m_model_path, triton_client::CUDASHM)) {
        return false;
    }

    m_input_shapes  = m_triton_client->get_model_infer()->m_input_shape["images"];
    m_output_shapes = m_triton_client->get_model_infer()->m_output_shape["output0"];

    if (m_type == YoloType::V5 || m_type == YoloType::V3 || m_type == YoloType::V7) {
        m_normalize  = CUDA::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDA::ChannelType::Invert);
        m_class_nums = m_output_shapes[2] - 5;
    } else if (m_type == YoloType::V8) {
        m_normalize  = CUDA::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDA::ChannelType::Invert);
        m_class_nums = m_output_shapes[2] - 4;
    }  else if (m_type == YoloType::X) {
        float mean[] = {0.485, 0.456, 0.406};
        float std[]  = {0.229, 0.224, 0.225};
        m_normalize  = CUDA::Norm::mean_std(mean, std, 1 / 255.0f, CUDA::ChannelType::Invert);
        m_normalize  = CUDA::Norm::None();
        m_class_nums = m_output_shapes[2] - 5;
    } else {
        ErrorL << "not support type";
        return false;
    }
    return true;
}

void YoloDetectionInfer::pre_process(Data::BatchData::ptr &data) {
//    TimeTicker();
    int batch_size = data->batch_data.size();
    DebugL << "batch_size: " << batch_size;
    data->BATCH_INPUT_TENSOR = m_tensor_allocator->query();
    data->BATCH_OUTPUT_TENSOR = m_tensor_allocator->query();
    auto &input_tensor = data->BATCH_INPUT_TENSOR->data();
    auto &output_tensor = data->BATCH_OUTPUT_TENSOR->data();
    if(!input_tensor){
        input_tensor = std::make_shared<CUDA::Tensor>();
        input_tensor->set_workspace(std::make_shared<CUDA::MixMemory>());
        CUDA::CUStream stream;
        checkCudaRuntime(cudaStreamCreate(&stream));
        input_tensor->set_stream(stream, true);
    }
    if(!output_tensor){
        output_tensor = std::make_shared<CUDA::Tensor>();
    }
    input_tensor->resize(batch_size, m_input_shapes[1], m_input_shapes[2], m_input_shapes[3]);
    output_tensor->resize(batch_size, m_output_shapes[1], m_output_shapes[2]);
    for (int i = 0; i < batch_size; ++i) {
        image_to_tensor(data->batch_data[i], input_tensor, i);
    }
}

void YoloDetectionInfer::infer_process(Data::BatchData::ptr &batch_data) {
    TimeTicker();
    m_triton_client->AddInput("images", batch_data->BATCH_INPUT_TENSOR->data());
    m_triton_client->infer();
    m_triton_client->GetOutput("output0", batch_data->BATCH_OUTPUT_TENSOR->data());
}

void YoloDetectionInfer::post_process(Data::BatchData::ptr &batch_data) {
//    TimeTicker();
    int          batch_size = batch_data->batch_data.size();
    CUDA::CUStream stream = batch_data->BATCH_INPUT_TENSOR->data()->get_stream();
    CUDA::Tensor output_array_device(CUDA::DataType::FP32);
    output_array_device.resize(batch_size, 32 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();
    for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
        float *image_based_output = batch_data->BATCH_OUTPUT_TENSOR->data()->gpu<float>(ibatch);
        float *output_array_ptr   = output_array_device.gpu<float>(ibatch);
        auto   affine_matrix      = batch_data->batch_data[ibatch]
                                 ->CUDA_AFFINMATRIX_TENSOR
                                 ->gpu<float>();
        checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream));
        if (m_type == YoloType::V8 ) {
            CUDA::decode_detect_yolov8_kernel_invoker(image_based_output, m_output_shapes[1],
                                                      m_class_nums, m_output_shapes[2],
                                                      m_confidence_threshold, affine_matrix,
                                                      output_array_ptr, MAX_IMAGE_BBOX, stream);
        } else {
            CUDA::decode_kernel_common_invoker(image_based_output, m_output_shapes[1], m_class_nums,
                                               m_output_shapes[2], m_confidence_threshold,
                                               affine_matrix, output_array_ptr, MAX_IMAGE_BBOX,
                                               stream);
        }
        CUDA::nms_kernel_invoker(output_array_ptr, m_nms_threshold, MAX_IMAGE_BBOX, stream);
    }
    // 同步
    output_array_device.to_cpu();

    for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
        float         *parray = output_array_device.cpu<float>(ibatch);
        int            count  = std::min(MAX_IMAGE_BBOX, (int)*parray);
        DetectBoxArray boxArray;
        for (int i = 0; i < count; ++i) {
            float *pbox     = parray + 1 + i * NUM_BOX_ELEMENT;
            int    label    = pbox[5];
            int    keepflag = pbox[6];
            if (keepflag == 1) {
                // x y w h conf label
                DetectBox box  = {pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label};
                box.class_name = m_class_names[label];
                boxArray.emplace_back(box);
            }
        }
        batch_data->batch_data[ibatch]->DETECTBOX_PROMISE->set_value(boxArray);
    }
    batch_data->BATCH_INPUT_TENSOR->release();
    batch_data->BATCH_OUTPUT_TENSOR->release();
}

void YoloDetectionInfer::image_to_tensor(Data::BaseData::ptr           &data,
                                         std::shared_ptr<CUDA::Tensor> &tensor,
                                         int                            ibatch) {
    auto& image = data->MAT_IMAGE;
    cudaSetDevice(m_device_id);
    cv::Size                input_size(tensor->size(3), tensor->size(2));
    CUDATools::AffineMatrix affin_matrix;
    affin_matrix.compute(image.size(), input_size);
    // 存放affin_matrix的tensor
    auto affin_matrix_tensor = std::make_shared<CUDA::Tensor>();
    affin_matrix_tensor->resize(sizeof(affin_matrix.d2i) * 2);
    data->CUDA_AFFINMATRIX_TENSOR=affin_matrix_tensor;
    data->CUDA_AFFINMATRIX=affin_matrix;

    // workspace地址空间有问题 需要修复
    size_t   size_image    = image.cols * image.rows * 3;
    size_t   size_matrix   = iLogger::upbound(sizeof(affin_matrix.d2i), 32);
    auto     workspace     = tensor->get_workspace();
    uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(size_matrix + size_image);
    //    float   *affine_matrix_device = (float *)gpu_workspace;
    uint8_t *image_device = size_matrix + gpu_workspace;

    uint8_t *cpu_workspace      = (uint8_t *)workspace->cpu(size_matrix + size_image);
    float   *affine_matrix_host = (float *)cpu_workspace;
    uint8_t *image_host         = size_matrix + cpu_workspace;
    auto     stream             = tensor->get_stream();

    memcpy(image_host, image.data, size_image);
    memcpy(affine_matrix_host, affin_matrix.d2i, sizeof(affin_matrix.d2i));
    checkCudaRuntime(
        cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));

    checkCudaRuntime(cudaMemcpyAsync(affin_matrix_tensor->gpu(), affine_matrix_host,
                                     sizeof(affin_matrix.d2i), cudaMemcpyHostToDevice, stream));

    //    checkCudaRuntime(cudaMemcpyAsync(affin_matrix_tensor->gpu(), affine_matrix_device,
    //                                     sizeof(affin_matrix.d2i), cudaMemcpyDeviceToDevice));

    CUDA::warpAffineBilinearAndNormalizePlaneInvoker(image_device, image.cols * 3, image.cols,
                                                     image.rows, tensor->gpu<float>(ibatch),
                                                     input_size.width, input_size.height,
                                                     affin_matrix_tensor->gpu<float>(), 114,
                                                     m_normalize, stream);

    //    tensor->synchronize();
}

Data::BaseData::ptr YoloDetectionInfer::commit(const Data::BaseData::ptr &data) {
    // 如果数据添加成功，返回一个promise，否则返回nullptr
    bool res = m_infer_node->add_data_back(data);
    if (res) {
        std::shared_ptr<std::promise<DetectBoxArray>> box_array_promise =
            std::make_shared<std::promise<DetectBoxArray>>();
        std::shared_future<DetectBoxArray> box_array_future = box_array_promise->get_future();
        data->DETECTBOX_FUTURE=box_array_future;
        data->DETECTBOX_PROMISE=box_array_promise;
        return data;
    } else{
        ErrorL << "数据添加失败";
        ErrorL << "id: " << data->FRAME_INDEX;
    }
    return nullptr;
}

}  // namespace infer