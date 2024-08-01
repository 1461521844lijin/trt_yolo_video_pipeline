//
// Created by Lenovo on 24-7-30.
//

#ifndef VIDEOPIPELINE_TRITONSERVER_H
#define VIDEOPIPELINE_TRITONSERVER_H

#include "cuda_kernels/cuda_tools/Tensor.h"
#include "triton/core/tritonserver.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#define TRITON_SERVER_DEFAULT_INIT(model_repository_path, verbose_level)                           \
    triton_server::TritonServer::Instance().init(model_repository_path, verbose_level)

#define TRITON_SERVER_CUSTOM_INIT(model_repository_path, verbose_level, backends_path,             \
                                  repo_agent_path)                                                 \
    triton_server::TritonServer::Instance().init(model_repository_path, verbose_level,             \
                                                 backends_path, repo_agent_path)

#define TRITON_SERVER_GET_MODEL_PLATFORM(model_name, model_version)                                \
    triton_server::TritonServer::Instance().getModelPlatform(model_name, model_version)

#define TRITON_SERVER_INFER(model_name, model_version, inputs, outputs)                            \
    triton_server::TritonServer::Instance().infer(model_name, model_version, inputs, outputs)

#define TRITON_SERVER_UNINIT() triton_server::TritonServer::Instance().uninit()

namespace triton_server {

class TritonTensor {
public:
    typedef std::shared_ptr<TritonTensor>    ptr;
    std::map<std::string, CUDA::Tensor::ptr> tensors;  // name -> tensor
};

typedef struct ModelInfo {
    std::string                                  name;              // 模型名称
    std::string                                  version;           // 模型版本
    std::string                                  platform;          // 模型平台
    std::string                                  model_key;         // 模型key
    int                                          max_batch_size;    // 最大批处理大小
    int                                          input_count;       // 输入数量
    int                                          output_count;      // 输出数量
    std::map<std::string, TRITONSERVER_DataType> inputs_datatype;   // 输入数据类型
    std::map<std::string, TRITONSERVER_DataType> outputs_datatype;  // 输出数据类型
    std::map<std::string, std::vector<int64_t>>  inputs_dims;       // 输入shape
    std::map<std::string, std::vector<int64_t>>  outputs_dims;      // 输出shape
} ModelInfo;

class TritonServer {
public:
    typedef std::shared_ptr<TritonServer> ptr;

public:
    /**
     * 获取单例
     * @return
     */
    static TritonServer &Instance();

    /**
     * 初始化
     * @param model_repository_path 模型仓库路径
     * @param verbose_level     日志级别
     * @param backends_path     后端路径
     * @param repo_agent_path   代理路径
     * @param timeout           超时时间
     */
    void init(const std::string &model_repository_path,
              int                verbose_level   = 0,
              const std::string &backends_path   = "/opt/tritonserver/backends",
              const std::string &repo_agent_path = "/opt/tritonserver/repoagents",
              int                timeout         = 500);
    /**
     * 反初始化
     */
    void uninit();

    void infer(const std::string                   &model_name,
               const std::string                   &model_version,
               const std::shared_ptr<TritonTensor> &input_tensor,
               std::shared_ptr<TritonTensor>       &output_tensor);

    /**
     * 获取模型config的json字符串
     * @param model_name    模型名称
     * @param model_version 模型版本
     * @return
     */
    std::string get_model_config(const std::string &model_name, const std::string &model_version);

    /**
     * 获取模型metadata的json字符串
     * @param model_name    模型名称
     * @param model_version 模型版本
     * @return
     */
    std::string get_model_metadata(const std::string &model_name, const std::string &model_version);

    /**
     * 获取模型平台
     * @param model_name    模型名称
     * @param model_version 模型版本
     * @return
     */
    std::string get_model_platform(const std::string &model_name, const std::string &model_version);

private:
    TritonServer() = default;
    ~TritonServer();

    /**
     * 解析模型推理响应，获取输出tensor
     * @param response
     * @param model_info
     * @param output_tensor
     */
    static void parseModelInferResponse(TRITONSERVER_InferenceResponse *response,
                                        ModelInfo                      &model_info,
                                        std::shared_ptr<TritonTensor>  &output_tensor);

private:
    uint32_t                             m_api_version_major;      // api版本
    uint32_t                             m_api_version_minor;      // api版本
    std::string                          m_model_repository_path;  // 模型仓库路径
    int32_t                              m_verbose_level;          // 日志级别
    std::map<std::string, ModelInfo>     m_models_info;            // 模型信息
    std::shared_ptr<TRITONSERVER_Server> m_server;                 // triton server
};

}  // namespace triton_server

#endif  // VIDEOPIPELINE_TRITONSERVER_H
