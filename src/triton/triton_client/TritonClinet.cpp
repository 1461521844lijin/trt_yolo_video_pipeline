//
// Created by lijin on 24-7-9.
//

#include "TritonClinet.h"

namespace triton_client {

void TritonClient::add_http_header(const std::string &key, const std::string &value) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_http_headers[key] = value;
}
bool TritonClient::Infer(TritonModelInfer::ptr model_infer) {
    InfoL << "TODO";
    return false;
}
bool TritonClient::InitSharedMemory(TritonModelInfer::ptr model_infer) {
    InfoL << "TODO";
    return false;
}
bool TritonClient::InitCudaSharedMemory(TritonModelInfer::ptr model_infer) {
    return false;
}
}  // namespace triton_client