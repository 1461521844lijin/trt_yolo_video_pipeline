//
// Created by lijin on 24-7-18.
//

#ifndef VIDEOPIPELINE_TRITONMODELINFER_H
#define VIDEOPIPELINE_TRITONMODELINFER_H

#include "common.h"
#include "cuda_kernels/cuda_tools/Tensor.h"
#include "grpc_client.h"
#include "http_client.h"

namespace triton_client {
namespace tc = triton::client;

// 数据传输模式  网络模式、系统共享内存、cuda共享显存
enum DateTransMode {
    NET     = 0,
    SHM     = 1,
    CUDASHM = 2,
};

/*
 * 客户端调用推理时的封装对象，内部包含了推理的输入输出信息
 */
class TritonModelInfer {
public:
    typedef std::shared_ptr<TritonModelInfer> ptr;

    ~TritonModelInfer();

public:
    /**
     *  对模型输入头添加输入
     *  @param input_name 输入名称
     *  @param tensor 输入数据
     *  @return
     */
    bool AddInput(const std::string &input_name, const CUDA::Tensor::ptr &tensor);

    /**
     * 获取模型输出
     * @param output_name 输出名称
     * @param output 输出数据 <std::string, tensor>
     * @return
     */
    bool GetOutput(const std::string &output_name, CUDA::Tensor::ptr &output);

public:
    /**
     * 解析模型配置
     * @param model_metadata
     * @param model_config
     * @return
     */
    virtual bool ParseModelGrpcConfig(const inference::ModelMetadataResponse &model_metadata,
                                      const inference::ModelConfigResponse   &model_config);

    /**
     * 解析模型配置
     * @param model_metadata
     * @param model_config
     * @return
     */
    virtual bool ParseModelHttpConfig(const std::string &model_metadata,
                                      const std::string &model_config);

public:
    std::string      m_model_name;          // 模型名称
    std::string      m_model_version;       // 模型版本
    int              m_max_batch_size = 1;  // 最大批处理大小
    tc::InferOptions m_infer_options  = tc::InferOptions("");
    DateTransMode    m_data_mode      = DateTransMode::NET;

    // 输入相关
    std::map<std::string, std::shared_ptr<tc::InferInput>> m_inputs_map;      // 客户端输入map
    std::map<std::string, std::vector<int64_t>>            m_input_shape;     // 输入shape
    std::map<std::string, std::string>                     m_input_datatype;  // 输入数据类型
    std::map<std::string, size_t> m_input_byte_sizes;                         // 输入数据大小
    std::map<std::string, size_t> m_input_byte_offsets;                       // 输入数据偏移
    size_t                        m_total_input_byte_size = 0;  // 输入数据总大小
    std::map<std::string, void *> m_input_shm_ptr;              // 系统输入共享内存指针
    void                         *m_input_shm_addr = nullptr;   // 系统输入共享内存地址
    int                           m_shm_input_fd   = -1;  // 系统输入共享内存文件描述符
    std::string                   m_input_shm_key;        // 系统共享内存key
    std::string                   m_input_region_name;    // 输入区域名称
    std::map<std::string, void *> m_input_cudashm_ptr;    // 输入cuda共享内存指针
    void                         *m_input_cudashm_addr = nullptr;  // cuda共享内存指针
    cudaIpcMemHandle_t            m_input_cuda_handle{};           // cuda共享内存句柄

    // 输出相关
    std::map<std::string, std::shared_ptr<tc::InferRequestedOutput>>
                                                m_outputs_map;          // 客户端输出map
    std::map<std::string, std::vector<int64_t>> m_output_shape;         // 输出shape
    std::map<std::string, std::string>          m_output_datatype;      // 输出数据类型
    std::map<std::string, size_t>               m_output_byte_sizes;    // 输出数据大小
    std::map<std::string, size_t>               m_output_byte_offsets;  // 输出数据偏移
    size_t                        m_total_output_byte_size = 0;         // 输出数据总大小
    std::map<std::string, void *> m_output_shm_ptr;             // 系统输出共享内存指针
    void                         *m_output_shm_addr = nullptr;  // 系统输出共享内存地址
    int                           m_shm_output_fd = -1;  // 系统输出共享内存文件描述符
    std::string                   m_output_shm_key;      // 系统共享内存key
    std::string                   m_output_region_name;  // 输出区域名称
    std::map<std::string, void *> m_output_cudashm_ptr;  // 输出cuda共享内存指针
    void                         *m_output_cudashm_addr = nullptr;  // cuda共享内存指针
    cudaIpcMemHandle_t            m_output_cuda_handle{};           // cuda共享内存句柄

    std::vector<tc::InferInput *>                 m_inputs;   // 输出指针存放列表
    std::vector<const tc::InferRequestedOutput *> m_outputs;  // 输出指针存放列表
    std::shared_ptr<tc::InferResult>              m_result;   // 推理结果
};

}  // namespace triton_client

#endif  // VIDEOPIPELINE_TRITONMODELINFER_H
