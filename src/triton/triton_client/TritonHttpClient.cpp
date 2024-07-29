//
// Created by lijin on 24-7-12.
//

#include "TritonHttpClient.h"

namespace triton_client {
bool TritonHttpClient::init_client(const std::string &url, bool verbose) {
    m_url     = url;
    m_verbose = verbose;
    FAIL_IF_ERR(tc::InferenceServerHttpClient::Create(&m_client, url, verbose),
                "创建http client失败");
    return true;
}
std::string TritonHttpClient::get_repository_index() {
    std::string repository_index;
    RETURN_NULLSTR_CHECK(m_client->ModelRepositoryIndex(&repository_index),
                         "获取repository index失败");
    return repository_index;
}
void TritonHttpClient::load_model(const std::string &model_name) {
    FAIL_IF_ERR(m_client->LoadModel(model_name), "加载模型失败");
}
void TritonHttpClient::unload_model(const std::string &model_name) {
    FAIL_IF_ERR(m_client->UnloadModel(model_name), "卸载模型失败");
}
std::string TritonHttpClient::get_model_config(const std::string &model_name) {
    std::string model_config;
    RETURN_NULLSTR_CHECK(m_client->ModelConfig(&model_config, model_name), "获取模型配置失败");
    return model_config;
}
std::string TritonHttpClient::get_model_metadata(const std::string &model_name) {
    std::string model_metadata;
    RETURN_NULLSTR_CHECK(m_client->ModelMetadata(&model_metadata, model_name),
                         "获取模型metadata失败");
    return model_metadata;
}

bool TritonHttpClient::Infer(TritonModelInfer::ptr model_infer) {
    std::unique_lock<std::mutex> lock(m_mutex);
    tc::InferResult                              *m_result;   // 推理结果
    RETURN_FALSE_CHECK(m_client->Infer(&m_result, model_infer->m_infer_options,
                                       model_infer->m_inputs, model_infer->m_outputs,
                                       m_http_headers),
                       "推理失败")
    model_infer->m_result.reset(m_result);
    return true;
}
TritonModelInfer::ptr TritonHttpClient::CreateModelInfer(const std::string          &model_name,
                                                         DateTransMode mode) {
    auto model_config       = get_model_config(model_name);
    auto model_metadata     = get_model_metadata(model_name);
    if(model_config.empty() || model_metadata.empty()){
        return nullptr;
    }
    switch (mode) {
        case DateTransMode::NET: {
            TritonModelInfer::ptr model_infer = std::make_shared<TritonModelInfer>();
            model_infer->m_model_name         = model_name;
            if (!model_infer->ParseModelHttpConfig(model_metadata, model_config)) {
                ErrorL << "error: 解析模型配置失败" << std::endl;
                return nullptr;
            }
            return model_infer;
        }
        case SHM: {
            break;
        }
        case CUDASHM: {
            break;
        }
        default: return nullptr;
    }
}

}  // namespace triton_client