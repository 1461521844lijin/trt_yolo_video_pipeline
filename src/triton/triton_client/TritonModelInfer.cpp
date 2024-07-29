//
// Created by lijin on 24-7-18.
//

#include "TritonModelInfer.h"
#include "TritonClinet.h"
#include "grpc_service.pb.h"
#include "utils/logger.h"

namespace triton_client {

TritonModelInfer::~TritonModelInfer() {}

bool TritonModelInfer::ParseModelGrpcConfig(const inference::ModelMetadataResponse &model_metadata,
                                            const inference::ModelConfigResponse   &model_config) {
    InfoL << "解析grpc模型配置" << std::endl;
    InfoL << "模型输入数量: " << model_metadata.inputs_size() << std::endl;
    InfoL << "模型输出数量: " << model_metadata.outputs_size() << std::endl;
    m_max_batch_size = model_config.config().max_batch_size();
    if (m_max_batch_size < 1) {
        m_max_batch_size = 1;
    }
    InfoL << "模型最大批处理大小: " << m_max_batch_size << std::endl;
    for (int i = 0; i < model_metadata.inputs_size(); i++) {
        const auto &input_metadata = model_metadata.inputs(0);
        const auto &input_config   = model_config.config().input(i);
        InfoL << "输入名称: " << input_metadata.name() << std::endl;
        InfoL << "输入数据类型: " << input_metadata.datatype() << std::endl;
        InfoL << "输入形状: ";
        std::vector<int64_t> shape;
        for (int j = 0; j < input_metadata.shape().size(); j++) {
            InfoL << input_metadata.shape(j) << " ";
            shape.push_back(input_metadata.shape(j));
        }
        if (shape[0] == -1) {
            shape[0] = m_max_batch_size;
        }
        tc::InferInput *input;
        FAIL_IF_ERR(tc::InferInput::Create(&input, input_metadata.name(), shape,
                                           input_metadata.datatype()),
                    "创建输入失败");
        /// 初始化输入
        std::shared_ptr<tc::InferInput> input_ptr_shared(input);
        m_inputs_map[input_metadata.name()]     = input_ptr_shared;
        m_input_shape[input_metadata.name()]    = shape;
        m_input_datatype[input_metadata.name()] = input_metadata.datatype();
        m_input_byte_sizes[input_metadata.name()] =
            CUDA::data_nums(shape) * CUDA::data_type_size(input_metadata.datatype());
        m_input_byte_offsets[input_metadata.name()] = m_total_input_byte_size;
        m_total_input_byte_size += m_input_byte_sizes[input_metadata.name()];
        m_inputs.push_back(input);
    }

    for (int i = 0; i < model_metadata.outputs_size(); i++) {
        const auto &output_metadata = model_metadata.outputs(0);
        const auto &output_config   = model_config.config().output(i);
        InfoL << "输出名称: " << output_metadata.name() << std::endl;
        InfoL << "输出数据类型: " << output_metadata.datatype() << std::endl;
        InfoL << "输出形状: ";
        std::vector<int64_t> shape;
        for (int j = 0; j < output_metadata.shape().size(); j++) {
            InfoL << output_metadata.shape(j);
            shape.push_back(output_metadata.shape(j));
        }
        if (shape[0] == -1) {
            shape[0] = m_max_batch_size;
        }
        tc::InferRequestedOutput *output;
        FAIL_IF_ERR(tc::InferRequestedOutput::Create(&output, output_metadata.name()),
                    "创建输出失败");
        /// 初始化输出
        std::shared_ptr<tc::InferRequestedOutput> output_ptr_shared(output);
        m_outputs_map[output_metadata.name()]     = output_ptr_shared;
        m_output_shape[output_metadata.name()]    = shape;
        m_output_datatype[output_metadata.name()] = output_metadata.datatype();
        m_output_byte_sizes[output_metadata.name()] =
            CUDA::data_nums(shape) * CUDA::data_type_size(output_metadata.datatype());
        m_output_byte_offsets[output_metadata.name()] = m_total_output_byte_size;
        m_total_output_byte_size += m_output_byte_sizes[output_metadata.name()];
        m_outputs.push_back(output);
    }
    m_model_name                   = model_metadata.name();
    m_model_version                = model_metadata.versions().at(0);
    m_infer_options.model_name_    = model_metadata.name();
    m_infer_options.model_version_ = model_metadata.versions().at(0);
    return true;
}
bool TritonModelInfer::ParseModelHttpConfig(const std::string &model_metadata,
                                            const std::string &model_config) {
    InfoL << "TODO";
    return false;
}

#define FALSE_CHECK(status, msg)                                                                   \
    if (!(status)) {                                                                               \
        ErrorL << msg << std::endl;                                                                \
        return false;                                                                              \
    }

bool TritonModelInfer::AddInput(const std::string &input_name, const CUDA::Tensor::ptr &tensor) {
    // 1、先校验一下输入数据是否合法
    FALSE_CHECK(tensor, "error: 输入数据为空");
    FALSE_CHECK(tensor->bytes() != 0, "error: 输入数据为空");
    auto it = m_inputs_map.find(input_name);
    FALSE_CHECK(it != m_inputs_map.end(), "error: 输入名称不存在");
    auto input = it->second;
    FALSE_CHECK(input->Shape().size() == tensor->dims().size(), "error: 输入数据shape维度不匹配");
    // 实际的输入batch不能超过模型设置的最大batch
    FALSE_CHECK(tensor->dims().at(0) <= m_max_batch_size, "error: 输入batch大小超过模型最大batch");
    for (int i = 1; i < input->Shape().size(); i++) {
        FALSE_CHECK(input->Shape().at(i) == tensor->dims().at(i), "error: 输入数据shape大小不匹配");
    }
    // 2、以上条件都满足，重新设置输入的batch大小
    input->SetShape(tensor->dims());
    // 3、将输入数据拷贝到输入tensor中  cpu拷贝
    switch (m_data_mode) {
        case DateTransMode::NET: {
            FAIL_IF_ERR(input->AppendRaw(reinterpret_cast<uint8_t *>(tensor->cpu()),
                                         tensor->bytes()),
                        "设置输入数据失败");
            break;
        }
        case DateTransMode::SHM: {
            // 根据输入数据大小重新设置共享内存大小
            FAIL_IF_ERR(input->SetSharedMemory(m_input_region_name, tensor->bytes(),
                                               m_input_byte_offsets[input_name]),
                        "设置输入数据失败");
            // 拷贝数据到共享内存
            memcpy(m_input_shm_ptr[input_name], tensor->cpu(), tensor->bytes());
            break;
        }
        case DateTransMode::CUDASHM: {
            // 根据输入数据大小重新设置共享显存大小
            FAIL_IF_ERR(input->SetSharedMemory(m_input_region_name, tensor->bytes(),
                                               m_input_byte_offsets[input_name]),
                        "设置输入数据失败");
            // 拷贝数据到共享显存
            cudaMemcpy(m_input_cudashm_ptr[input_name], tensor->gpu(), tensor->bytes(),
                       cudaMemcpyDeviceToDevice);
            break;
        }
    }

    return true;
}
bool TritonModelInfer::GetOutput(const std::string &output_name, CUDA::Tensor::ptr &tensor) {
    // 校验数据
    RETURN_FALSE_CHECK(m_result->RequestStatus(), "推理失败");
    auto it = m_outputs_map.find(output_name);
    FALSE_CHECK(it != m_outputs_map.end(), "error: 输出名称不存在");
    std::vector<int64_t> output_shape;
    RETURN_FALSE_CHECK(m_result->Shape(output_name, &output_shape), "获取输出shape失败");
//    DebugL << "输出shape: ";
//    for (auto &i : output_shape) {
//        DebugL << i << " ";
//    }
    std::string output_datatype;
    RETURN_FALSE_CHECK(m_result->Datatype(output_name, &output_datatype), "获取输出数据类型失败");
//    DebugL << "输出数据类型: " << output_datatype << std::endl;
    FALSE_CHECK(output_datatype == m_output_datatype[output_name], "error: 输出数据类型不匹配");

    // 拷贝输出数据
    if (!tensor) {
        tensor =
            std::make_shared<CUDA::Tensor>(output_shape, CUDA::string_to_type(output_datatype));
    }
    tensor->resize(output_shape);
    switch (m_data_mode) {
        case DateTransMode::NET: {
            size_t output_byte_size;
            float *data_ptr;
            m_result->RawData(output_name, (const uint8_t **)&data_ptr, &output_byte_size);
            DebugL << "输出bytes大小: " << output_byte_size << std::endl;
            // 拷贝数据到tensor
            memcpy(tensor->cpu(), data_ptr, output_byte_size);
            data_ptr = nullptr;
            break;
        }
        case DateTransMode::SHM: {
            // 拷贝共享内存中的数据到tensor
            memcpy(tensor->cpu(), m_output_shm_ptr[output_name], tensor->bytes());
            break;
        }
        case DateTransMode::CUDASHM: {
            // 拷贝共享显存中的数据到tensor
            cudaMemcpy(tensor->gpu(), m_output_cudashm_ptr[output_name], tensor->bytes(),
                       cudaMemcpyDeviceToDevice);
            break;
        }

    }

    return true;
}

}  // namespace triton_client