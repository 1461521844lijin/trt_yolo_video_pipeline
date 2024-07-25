//
// Created by Lenovo on 24-7-9.
//

#ifndef VIDEOPIPELINE_TRITONGRPCCLIENT_H
#define VIDEOPIPELINE_TRITONGRPCCLIENT_H

#include "TritonClinet.h"
#include "TritonModelInfer.h"
#include "grpc_client.h"

namespace triton_client {

class TritonGrpcClient : public TritonClient {
public:
    typedef std::shared_ptr<TritonGrpcClient> ptr;

    ~TritonGrpcClient();

public:
    bool init_client(const std::string &url, bool verbose);

    // 获取repository index
    std::string get_repository_index();

    void load_model(const std::string &model_name) override;

    void unload_model(const std::string &model_name) override;

    bool IsModelReady(const std::string &model_name) override;

    bool infer();

    /** 获取推理对象
     * @return
     */
    TritonModelInfer::ptr get_model_infer() const;

    bool init_model_infer(const std::string &model_name, DateTransMode mode);

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

private:
    bool Infer(TritonModelInfer::ptr model_infer) override;

    TritonModelInfer::ptr CreateModelInfer(const std::string &model_name,
                                           DateTransMode      mode) override;

    bool InitSharedMemory(TritonModelInfer::ptr model_infer) override;

    bool InitCudaSharedMemory(TritonModelInfer::ptr model_infer) override;

    bool UnInitSharedMemory(const TritonModelInfer::ptr &model_infer);

    bool UnInitCudaSharedMemory(const TritonModelInfer::ptr &model_infer);

private:
    std::unique_ptr<tc::InferenceServerGrpcClient> m_client      = nullptr;
    std::shared_ptr<TritonModelInfer>              m_model_infer = nullptr;

};

}  // namespace triton_client

#endif  // VIDEOPIPELINE_TRITONGRPCCLIENT_H
