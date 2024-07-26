//
// Created by lijin on 24-7-9.
//

#include "TritonGrpcClient.h"
#include "grpc_service.pb.h"
#include "shm_utils.h"
#include "utils/TimeTicker.h"

#define FAIL_IF_CUDA_ERR(FUNC)                                                                     \
    {                                                                                              \
        const cudaError_t result = FUNC;                                                           \
        if (result != cudaSuccess) {                                                               \
            std::cerr << "CUDA exception (line " << __LINE__ << "): " << cudaGetErrorName(result)  \
                      << " (" << cudaGetErrorString(result) << ")" << std::endl;                   \
            exit(1);                                                                               \
        }                                                                                          \
    }

namespace triton_client {
bool TritonGrpcClient::init_client(const std::string &url, bool verbose) {
    m_url = url;
    FAIL_IF_ERR(tc::InferenceServerGrpcClient::Create(&m_client, url, verbose),
                "创建grpc client失败")
    return true;
}
std::string TritonGrpcClient::get_repository_index() {
    inference::RepositoryIndexResponse repository_index;
    FAIL_IF_ERR(m_client->ModelRepositoryIndex(&repository_index), "获取repository index失败")
    return repository_index.DebugString();
}
void TritonGrpcClient::load_model(const std::string &model_name) {
    FAIL_IF_ERR(m_client->LoadModel(model_name), "加载模型失败")
}
void TritonGrpcClient::unload_model(const std::string &model_name) {
    FAIL_IF_ERR(m_client->UnloadModel(model_name), "卸载模型失败")
}
bool TritonGrpcClient::IsModelReady(const std::string &model_name) {
    bool model_ready;
    FAIL_IF_ERR(m_client->IsModelReady(&model_ready, model_name), "获取模型状态失败")
    return model_ready;
}
TritonModelInfer::ptr TritonGrpcClient::CreateModelInfer(const std::string &model_name,
                                                         DateTransMode      mode) {
    if (!IsModelReady(model_name)) {
        ErrorL << "error: 服务端模型未加载" << std::endl;
        return nullptr;
    }
    TritonModelInfer::ptr model_infer = std::make_shared<TritonModelInfer>();
    model_infer->m_model_name         = model_name;
    inference::ModelMetadataResponse model_metadata;
    RETURN_NULL_CHECK(m_client->ModelMetadata(&model_metadata, model_name), "获取模型metadata失败")
    inference::ModelConfigResponse model_config;
    RETURN_NULL_CHECK(m_client->ModelConfig(&model_config, model_name), "获取模型配置失败")
    if (!model_infer->ParseModelGrpcConfig(model_metadata, model_config)) {
        ErrorL << "error: 解析模型配置失败" << std::endl;
        return nullptr;
    }
    if (mode == SHM) {
        if (!InitSharedMemory(model_infer)) {
            ErrorL << "error: 初始化共享内存失败" << std::endl;
            return nullptr;
        }
    } else if (mode == CUDASHM) {
        if (!InitCudaSharedMemory(model_infer)) {
            ErrorL << "error: 初始化共享内存失败" << std::endl;
            return nullptr;
        }
    }
    return model_infer;
}

bool TritonGrpcClient::Infer(TritonModelInfer::ptr model_infer) {
//    TimeTicker();
    std::unique_lock<std::mutex> lock(m_mutex);
    RETURN_FALSE_CHECK(m_client->Infer(&model_infer->m_result, model_infer->m_infer_options,
                                       model_infer->m_inputs, model_infer->m_outputs,
                                       m_http_headers),
                       "推理失败")
    return true;
}

bool TritonGrpcClient::InitSharedMemory(TritonModelInfer::ptr model_infer) {
    // 1、注销之前的共享内存
    model_infer->m_input_region_name =
        model_infer->m_model_name + "_input_data_" + std::to_string(getpid());
    model_infer->m_output_region_name =
        model_infer->m_model_name + "_output_data_" + std::to_string(getpid());
    m_client->UnregisterSystemSharedMemory(model_infer->m_input_region_name);
    m_client->UnregisterSystemSharedMemory(model_infer->m_output_region_name);

    // 2、创建共享内存
    // 创建输入共享内存
    model_infer->m_input_shm_key = "/" + model_infer->m_model_name + "_input_shm";
    FAIL_IF_ERR(tc::CreateSharedMemoryRegion(model_infer->m_input_shm_key,
                                             model_infer->m_total_input_byte_size,
                                             &model_infer->m_shm_input_fd),
                "")
    // 映射输入共享内存
    FAIL_IF_ERR(tc::MapSharedMemory(model_infer->m_shm_input_fd, 0,
                                    model_infer->m_total_input_byte_size,
                                    &model_infer->m_input_shm_addr),
                "")
    FAIL_IF_ERR(tc::CloseSharedMemory(model_infer->m_shm_input_fd), "")
    FAIL_IF_ERR(m_client->RegisterSystemSharedMemory(model_infer->m_input_region_name,
                                                     model_infer->m_input_shm_key,
                                                     model_infer->m_total_input_byte_size),
                "")
    // 将输入共享内存地址传递给输入指针
    for (auto &[name, input] : model_infer->m_inputs_map) {
        model_infer->m_input_shm_ptr[name] =
            (uint8_t *)model_infer->m_input_shm_addr + model_infer->m_input_byte_offsets[name];
        FAIL_IF_ERR(input->SetSharedMemory(model_infer->m_input_region_name,
                                           model_infer->m_input_byte_sizes[name],
                                           model_infer->m_input_byte_offsets[name]),
                    "设置输入共享内存失败")
    }
    // 创建输出共享内存
    model_infer->m_output_shm_key = "/" + model_infer->m_model_name + "_output_shm";
    FAIL_IF_ERR(tc::CreateSharedMemoryRegion(model_infer->m_output_shm_key,
                                             model_infer->m_total_output_byte_size,
                                             &model_infer->m_shm_output_fd),
                "")
    // 映射输出共享内存
    FAIL_IF_ERR(tc::MapSharedMemory(model_infer->m_shm_output_fd, 0,
                                    model_infer->m_total_output_byte_size,
                                    &model_infer->m_output_shm_addr),
                "")
    FAIL_IF_ERR(tc::CloseSharedMemory(model_infer->m_shm_output_fd), "")
    FAIL_IF_ERR(m_client->RegisterSystemSharedMemory(model_infer->m_output_region_name,
                                                     model_infer->m_output_shm_key,
                                                     model_infer->m_total_output_byte_size),
                "")
    // 将输出共享内存地址传递给输出指针
    for (auto &[name, output] : model_infer->m_outputs_map) {
        model_infer->m_output_shm_ptr[name] =
            (uint8_t *)model_infer->m_output_shm_addr + model_infer->m_output_byte_offsets[name];
        FAIL_IF_ERR(output->SetSharedMemory(model_infer->m_output_region_name,
                                            model_infer->m_output_byte_sizes[name],
                                            model_infer->m_output_byte_offsets[name]),
                    "设置输出共享内存失败")
    }
    DebugL << "初始化共享内存成功" << std::endl;
    model_infer->m_data_mode = SHM;
    return true;
}

void CreateCUDAIPCHandle(cudaIpcMemHandle_t *cuda_handle, void *input_d_ptr, int device_id = 0) {
    // Set the GPU device to the desired GPU
    FAIL_IF_CUDA_ERR(cudaSetDevice(device_id))

    //  Create IPC handle for data on the gpu
    FAIL_IF_CUDA_ERR(cudaIpcGetMemHandle(cuda_handle, input_d_ptr))
}

bool TritonGrpcClient::InitCudaSharedMemory(TritonModelInfer::ptr model_infer) {
    // 1、注销之前的共享显存
    model_infer->m_input_region_name =
        model_infer->m_model_name + "_input_data_" + std::to_string(getpid());
    model_infer->m_output_region_name =
        model_infer->m_model_name + "_output_data_" + std::to_string(getpid());
    m_client->UnregisterCudaSharedMemory(model_infer->m_input_region_name);
    m_client->UnregisterCudaSharedMemory(model_infer->m_output_region_name);
    // 2、创建输入共享显存
    cudaMalloc(&model_infer->m_input_cudashm_addr, model_infer->m_total_input_byte_size);
    CreateCUDAIPCHandle(&model_infer->m_input_cuda_handle, model_infer->m_input_cudashm_addr);
    m_client->RegisterCudaSharedMemory(model_infer->m_input_region_name,
                                       model_infer->m_input_cuda_handle, 0 /* device_id */,
                                       model_infer->m_total_input_byte_size);
    for (auto &[name, input] : model_infer->m_inputs_map) {
        model_infer->m_input_cudashm_ptr[name] =
            (uint8_t *)model_infer->m_input_cudashm_addr + model_infer->m_input_byte_offsets[name];
        FAIL_IF_ERR(input->SetSharedMemory(model_infer->m_input_region_name,
                                           model_infer->m_input_byte_sizes[name],
                                           model_infer->m_input_byte_offsets[name]),
                    "设置输入共享显存失败")
    }

    cudaMalloc(&model_infer->m_output_cudashm_addr, model_infer->m_total_output_byte_size);
    CreateCUDAIPCHandle(&model_infer->m_output_cuda_handle, model_infer->m_output_cudashm_addr);
    m_client->RegisterCudaSharedMemory(model_infer->m_output_region_name,
                                       model_infer->m_output_cuda_handle, 0 /* device_id */,
                                       model_infer->m_total_output_byte_size);

    for (auto &[name, output] : model_infer->m_outputs_map) {
        model_infer->m_output_cudashm_ptr[name] = (uint8_t *)model_infer->m_output_cudashm_addr +
                                                  model_infer->m_output_byte_offsets[name];
        FAIL_IF_ERR(output->SetSharedMemory(model_infer->m_output_region_name,
                                            model_infer->m_output_byte_sizes[name],
                                            model_infer->m_output_byte_offsets[name]),
                    "设置输出共享显存失败")
    }
    DebugL << "初始化共享显存成功" << std::endl;
    model_infer->m_data_mode = CUDASHM;
    return true;
}

bool TritonGrpcClient::UnInitSharedMemory(const TritonModelInfer::ptr &model_infer) {
    if (model_infer->m_data_mode == SHM) {
        FAIL_IF_ERR(m_client->UnregisterSystemSharedMemory(model_infer->m_input_region_name), "")
        FAIL_IF_ERR(m_client->UnregisterSystemSharedMemory(model_infer->m_output_region_name), "")
        FAIL_IF_ERR(tc::UnmapSharedMemory(model_infer->m_input_shm_addr,
                                          model_infer->m_total_input_byte_size),
                    "")
        FAIL_IF_ERR(tc::UnmapSharedMemory(model_infer->m_output_shm_addr,
                                          model_infer->m_total_output_byte_size),
                    "")
        FAIL_IF_ERR(tc::UnlinkSharedMemoryRegion(model_infer->m_input_shm_key), "")
        FAIL_IF_ERR(tc::UnlinkSharedMemoryRegion(model_infer->m_output_shm_key), "")
        InfoL << "释放共享内存成功" << std::endl;
        return true;
    }
    return false;
}

bool TritonGrpcClient::UnInitCudaSharedMemory(const TritonModelInfer::ptr &model_infer) {
    if (model_infer->m_data_mode == CUDASHM) {
        FAIL_IF_ERR(m_client->UnregisterCudaSharedMemory(model_infer->m_input_region_name), "")
        FAIL_IF_ERR(m_client->UnregisterCudaSharedMemory(model_infer->m_output_region_name), "")
        FAIL_IF_CUDA_ERR(cudaFree(model_infer->m_input_cudashm_addr))
        FAIL_IF_CUDA_ERR(cudaFree(model_infer->m_output_cudashm_addr))
        InfoL << "释放共享显存成功" << std::endl;
        return true;
    }
    return false;
}

bool TritonGrpcClient::init_model_infer(const std::string &model_name, DateTransMode mode) {
    m_model_infer = CreateModelInfer(model_name, mode);
    if (!m_model_infer) {
        ErrorL << "error: 创建模型推理对象失败" << std::endl;
        return false;
    }
    return true;
}
bool TritonGrpcClient::AddInput(const std::string &input_name, const CUDA::Tensor::ptr &tensor) {
    if (!m_model_infer) {
        ErrorL << "error: 未初始化模型推理对象" << std::endl;
        return false;
    }
    return m_model_infer->AddInput(input_name, tensor);
}
bool TritonGrpcClient::GetOutput(const std::string &output_name, CUDA::Tensor::ptr &output) {
    if (!m_model_infer) {
        ErrorL << "error: 未初始化模型推理对象" << std::endl;
        return false;
    }
    return m_model_infer->GetOutput(output_name, output);
}
TritonGrpcClient::~TritonGrpcClient() {
    if (m_model_infer) {
        if (m_model_infer->m_data_mode == SHM) {
            UnInitSharedMemory(m_model_infer);
        } else if (m_model_infer->m_data_mode == CUDASHM) {
            UnInitCudaSharedMemory(m_model_infer);
        }
    }
}
bool TritonGrpcClient::infer() {
    return Infer(m_model_infer);
}
TritonModelInfer::ptr TritonGrpcClient::get_model_infer() const {
    return m_model_infer;
}

}  // namespace triton_client