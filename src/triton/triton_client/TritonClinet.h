//
// Created by lijin on 24-7-9.
//

#ifndef VIDEOPIPELINE_TRITONCLINET_H
#define VIDEOPIPELINE_TRITONCLINET_H

#include "TritonModelInfer.h"
#include "cuda_kernels/cuda_tools/Tensor.h"
#include "utils/logger.h"

#include <memory>


namespace triton_client {

#define FAIL_IF_ERR(X, MSG)                                                                        \
    {                                                                                              \
        tc::Error err = (X);                                                                       \
        if (!err.IsOk()) {                                                                         \
            ErrorL << "error: " << (MSG) << ": " << err << std::endl;                              \
            exit(1);                                                                               \
        }                                                                                          \
    }

#define RETURN_NULL_CHECK(X, MSG)                                                                  \
    {                                                                                              \
        tc::Error err = (X);                                                                       \
        if (!err.IsOk()) {                                                                         \
            ErrorL << "error: " << (MSG) << ": " << err << std::endl;                              \
            return nullptr;                                                                        \
        }                                                                                          \
    }

#define RETURN_FALSE_CHECK(X, MSG)                                                                 \
    {                                                                                              \
        tc::Error err = (X);                                                                       \
        if (!err.IsOk()) {                                                                         \
            ErrorL << "error: " << (MSG) << ": " << err << std::endl;                              \
            return false;                                                                          \
        }                                                                                          \
    }

#define RETURN_NULLSTR_CHECK(X, MSG)                                                               \
    {                                                                                              \
        tc::Error err = (X);                                                                       \
        if (!err.IsOk()) {                                                                         \
            ErrorL << "error: " << (MSG) << ": " << err << std::endl;                              \
            return "";                                                                             \
        }                                                                                          \
    }



class TritonClient {
public:
    typedef std::shared_ptr<TritonClient> ptr;



public:
    /**
     * 添加http头部信息
     * @param key
     * @param value
     */
    void add_http_header(const std::string &key, const std::string &value);

public:
    /**
     *  @param model_name 模型名称
     *  加载模型
     *  通过模型名称加载模型仓库只已有模型
     *  该函数只对server端启用了model_control_mode=explicit的情况可用
     */
    virtual void load_model(const std::string &model_name) = 0;

    /**
     * 卸载模型
     *  通过模型名称卸载模型仓库已有模型
     *  该函数只对server端启用了model_control_mode=explicit的情况可用
     *  @param model_name 模型名称
     *
     */
    virtual void unload_model(const std::string &model_name) = 0;

    /**
     * 模型是否已经加载
     * @param model_name
     * @return
     */
    virtual bool IsModelReady(const std::string &model_name) = 0;

public:
    /**
     * 创建模型推理对象
     * @param model_name 模型名称
     * @param mode 数据传输模式  网络模式、系统共享内存、cuda共享显存
     * @return TritonModelInfer::ptr
     */
    virtual TritonModelInfer::ptr
    CreateModelInfer(const std::string          &model_name,
                     DateTransMode mode) = 0;

    /**
     * 客户端推理调用
     *  @param model_infer 模型推理对象
     */
    virtual bool Infer(TritonModelInfer::ptr model_infer);

    /**
     * 初始化系统共享内存
     */
    virtual bool InitSharedMemory(TritonModelInfer::ptr model_infer);

    /**
     * 初始化CUDA共享内存
     */
    virtual bool InitCudaSharedMemory(TritonModelInfer::ptr model_infer);

protected:
    std::mutex  m_mutex;
    bool        m_verbose            = false;
    bool        m_client_initialized = false;
    std::string m_url;           // 服务器地址
    tc::Headers m_http_headers;  // http头部
};

}  // namespace triton_client

#endif  // VIDEOPIPELINE_TRITONCLINET_H
