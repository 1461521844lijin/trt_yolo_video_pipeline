//
// Created by lijin on 2023/12/21.
//

#include "YoloDetectionInfer.h"
#include "cuda_kernels/cuda_tools/cuda_tools.h"
#include "graph/core/common/DetectionBox.h"

#include <utility>

namespace infer {

static void affine_project(float *matrix, float x, float y, float *ox, float *oy) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

YoloDetectionInfer::YoloDetectionInfer(std::string infer_name,
                                       int         device_id,
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
    m_type = type;
}

bool YoloDetectionInfer::init() {
    m_trt_engine = TRT::TRTEngine::CreateShared(m_model_path, m_device_id);
    if (!m_trt_engine) {
        return false;
    }
    m_trt_engine->print();
    m_dynamic             = m_trt_engine->has_dynamic_dim();
    m_input_shapes        = m_trt_engine->static_dims(0);
    m_output_shapes       = m_trt_engine->static_dims(1);
    m_output_tensor       = std::make_shared<CUDA::Tensor>();
    m_affin_matrix_tensor = std::make_shared<CUDA::Tensor>();
    m_input_tensor        = std::make_shared<CUDA::Tensor>();
    m_segment_tensor      = std::make_shared<CUDA::Tensor>();
    m_input_tensor->set_workspace(std::make_shared<CUDA::MixMemory>());
    m_has_segmegt = m_type == YoloType::V8Seg;
    if (m_has_segmegt) {
        m_segment_shapes = m_trt_engine->static_dims(1);
        m_output_shapes  = m_trt_engine->static_dims(2);
    }
    if (m_type == YoloType::V5 || m_type == YoloType::V3 || m_type == YoloType::V7) {
        m_normalize  = CUDA::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDA::ChannelType::Invert);
        m_class_nums = m_output_shapes[2] - 5;
    } else if (m_type == YoloType::V8) {
        m_normalize  = CUDA::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDA::ChannelType::Invert);
        m_class_nums = m_output_shapes[2] - 4;
    } else if (m_type == YoloType::V8Seg) {
        m_normalize  = CUDA::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDA::ChannelType::Invert);
        m_class_nums = m_output_shapes[2] - 4 - m_segment_shapes[1];
    } else if (m_type == YoloType::X) {
        float mean[] = {0.485, 0.456, 0.406};
        float std[]  = {0.229, 0.224, 0.225};
        m_normalize  = CUDA::Norm::mean_std(mean, std, 1 / 255.0f, CUDA::ChannelType::Invert);
        m_normalize  = CUDA::Norm::None();
        m_class_nums = m_output_shapes[2] - 5;
    } else {
        std::cout << "not support type" << std::endl;
        return false;
    }
    return true;
}

void YoloDetectionInfer::pre_process(std::vector<Data::BaseData::ptr> &batch_data) {
    int batch_size = batch_data.size();
    m_input_tensor->resize(batch_size, m_input_shapes[1], m_input_shapes[2], m_input_shapes[3]);
    m_output_tensor->resize(batch_size, m_output_shapes[1], m_output_shapes[2]);
    for (int i = 0; i < batch_size; ++i) {
        auto &image = batch_data[i]->Get<MAT_IMAGE_TYPE>(MAT_IMAGE);
        image_to_tensor(image, m_input_tensor, i);
    }
}

void YoloDetectionInfer::infer_process(std::vector<Data::BaseData::ptr> &batch_data) {
    int infer_batch_size = batch_data.size();
    int model_batch_size = m_output_shapes[0];
    if (infer_batch_size !=
        model_batch_size) {  // 当模型单次推理batch_size不等于infer_batch时，需要重新设置模型的batch_size
        if (m_dynamic) {
            if (!m_trt_engine->set_run_dims(0, m_input_tensor->dims())) {
                std::cout << "设置推理batch失败！" << std::endl;
                batch_data.clear();
                return;
            }
        } else {
            if (infer_batch_size > model_batch_size) {
                std::cout << "静态模型的单次推理batch大小不能超过模型配置本身！" << std::endl;
                batch_data.clear();
                return;
            }
        }
    }
    std::vector<void *> bindings = {m_input_tensor->gpu(), m_output_tensor->gpu()};
    if (m_has_segmegt) {
        m_segment_tensor->resize(infer_batch_size, m_segment_shapes[1], m_segment_shapes[2],
                                 m_segment_shapes[3]);
        bindings = {m_input_tensor->gpu(), m_segment_tensor->gpu(), m_output_tensor->gpu()};
    }
    if (!m_trt_engine->forward(bindings, nullptr, nullptr)) {
        std::cout << "forward failed" << std::endl;
        batch_data.clear();
        return;
    }
}

void YoloDetectionInfer::post_process(std::vector<Data::BaseData::ptr> &batch_data) {
    int          batch_size = batch_data.size();
    CUDA::Tensor output_array_device(CUDA::DataType::Float);
    output_array_device.resize(batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();
    for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
        float *image_based_output = m_output_tensor->gpu<float>(ibatch);
        float *output_array_ptr   = output_array_device.gpu<float>(ibatch);
        auto   affine_matrix      = m_affin_matrix_tensor->gpu<float>();
        checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), m_stream));
        if (m_type == YoloType::V8 || m_type == YoloType::V8Seg) {
            CUDA::decode_detect_yolov8_kernel_invoker(image_based_output, m_output_shapes[1],
                                                      m_class_nums, m_output_shapes[2],
                                                      m_confidence_threshold, affine_matrix,
                                                      output_array_ptr, MAX_IMAGE_BBOX,
                                                      NUM_BOX_ELEMENT, m_stream);
        } else {
            CUDA::decode_kernel_common_invoker(image_based_output, m_output_shapes[1], m_class_nums,
                                               m_output_shapes[2], m_confidence_threshold,
                                               affine_matrix, output_array_ptr, MAX_IMAGE_BBOX,
                                               m_stream);
        }
        CUDA::nms_kernel_invoker(output_array_ptr, m_nms_threshold, MAX_IMAGE_BBOX, m_stream);
    }
    // 同步
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
                DetectBox box  = {pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label};
                box.class_name = m_class_names[label];

                /*
                if (m_has_segmegt) {
                    int    row_index = pbox[7];
                    float *mask_weights =
                        m_output_tensor->gpu<float>() +
                        (ibatch * m_output_shapes[1] + row_index) * m_output_shapes[2] +
                        m_class_nums + 4;

                    float left, top, right, bottom;
                    affine_project(m_affin_matrix.d2i, pbox[0], pbox[1], &left, &top);
                    affine_project(m_affin_matrix.d2i, pbox[2], pbox[3], &right, &bottom);
                    float scale_to_predict_x = m_segment_shapes[3] / (float)m_input_shapes[3];
                    float scale_to_predict_y = m_segment_shapes[2] / (float)m_input_shapes[2];
                    int   mask_out_width     = (right - left) * scale_to_predict_x + 0.5f;
                    int   mask_out_height    = (bottom - top) * scale_to_predict_y + 0.5f;

                    if (mask_out_width > 0 && mask_out_height > 0) {
                        if (!m_segment_tensor_cache) {  // 第一次进入初始化缓存
                            m_segment_tensor_cache = std::make_shared<CUDA::Tensor>();
                        }
                        int bytes_of_mask_out = mask_out_width * mask_out_height;
                        box.mask              = cv::Mat(mask_out_height, mask_out_width, CV_8UC1);
                        m_segment_tensor_cache->resize(bytes_of_mask_out);
                        CUDA::decode_single_mask(
                            left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                            m_segment_tensor->gpu<float>(ibatch), m_segment_shapes[3],
                            m_segment_shapes[2], m_segment_tensor_cache->gpu<unsigned char>(),
                            m_segment_shapes[1], mask_out_width, mask_out_height, m_stream);
                        checkCudaKernel(cudaMemcpyAsync(
                            box.mask.data, m_segment_tensor_cache->gpu<unsigned char>(),
                            m_segment_tensor_cache->bytes(), cudaMemcpyDeviceToHost, m_stream));
                        // 同步
                        checkCudaRuntime(cudaStreamSynchronize(m_stream));
                    }
                }
                 */

                boxArray.emplace_back(box);
            }
        }
        batch_data[ibatch]->Get<DETECTBOX_PROMISE_TYPE>(DETECTBOX_PROMISE)->set_value(boxArray);
    }
}

void YoloDetectionInfer::image_to_tensor(const cv::Mat                 &image,
                                         std::shared_ptr<CUDA::Tensor> &tensor,
                                         int                            ibatch) {
    cudaSetDevice(m_device_id);
    cv::Size input_size(tensor->size(3), tensor->size(2));
    m_affin_matrix.compute(image.size(), input_size);

    // workspace地址空间有问题 需要修复
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
    m_affin_matrix_tensor->resize(sizeof(m_affin_matrix.d2i));
    checkCudaRuntime(cudaMemcpyAsync(m_affin_matrix_tensor->gpu(), affine_matrix_device,
                                     sizeof(m_affin_matrix.d2i), cudaMemcpyDeviceToDevice));

    CUDA::warpAffineBilinearAndNormalizePlaneInvoker(
        image_device, image.cols * 3, image.cols, image.rows, tensor->gpu<float>(ibatch),
        input_size.width, input_size.height, affine_matrix_device, 114, m_normalize, stream);

    tensor->synchronize();
}

Data::BaseData::ptr YoloDetectionInfer::commit(const Data::BaseData::ptr &data) {
    std::shared_ptr<std::promise<DetectBoxArray>> box_array_promise =
        std::make_shared<std::promise<DetectBoxArray>>();
    std::shared_future<DetectBoxArray> box_array_future = box_array_promise->get_future();
    data->Insert<DETECTBOX_FUTURE_TYPE>(DETECTBOX_FUTURE, box_array_future);
    data->Insert<DETECTBOX_PROMISE_TYPE>(DETECTBOX_PROMISE, box_array_promise);
    m_infer_node->add_data(data);
    return data;
}

}  // namespace infer