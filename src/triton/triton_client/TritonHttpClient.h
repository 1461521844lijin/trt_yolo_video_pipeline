//
// Created by lijin on 24-7-12.
//

#ifndef VIDEOPIPELINE_TRITONHTTPCLIENT_H
#define VIDEOPIPELINE_TRITONHTTPCLIENT_H

#include "TritonClinet.h"
#include "http_client.h"

namespace triton_client {

/**
 * Triton http client
    暂不可用
 */
class TritonHttpClient : public TritonClient {
public:
    typedef std::shared_ptr<TritonHttpClient> ptr;

public:
    bool init_client(const std::string &url, bool verbose);

    // 获取repository index
    std::string get_repository_index();

    void load_model(const std::string &model_name) override;

    void unload_model(const std::string &model_name) override;

    std::string get_model_config(const std::string &model_name);

    std::string get_model_metadata(const std::string &model_name);

public:
    TritonModelInfer::ptr CreateModelInfer(const std::string &model_name,
                                           DateTransMode      mode) override;

    bool Infer(TritonModelInfer::ptr model_infer) override;

private:
    std::unique_ptr<tc::InferenceServerHttpClient> m_client = nullptr;
};

}  // namespace triton_client

#endif  // VIDEOPIPELINE_TRITONHTTPCLIENT_H
