//
// Created by Lenovo on 24-7-31.
//

#include "triton/triton_server/TritonServer.h"
#include "utils/logger.h"
#include <iostream>

int main() {
    toolkit::Logger::Instance().setLevel(toolkit::LInfo);
    std::string model_repository_path = "/models";
    std::string model_name            = "yolov8_fp16_trt";
    std::string model_version         = "1";

    int verbose_level = 0;
    TRITON_SERVER_DEFAULT_INIT(model_repository_path, verbose_level);

    auto                 inputs      = std::make_shared<triton_server::TritonTensor>();
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    inputs->tensors["images"]        = std::make_shared<CUDA::Tensor>();
    inputs->tensors["images"]->resize(input_shape).set_to(2);

    auto outputs                = std::make_shared<triton_server::TritonTensor>();
    outputs->tensors["output0"] = std::make_shared<CUDA::Tensor>();

    for (int i = 0; i < 1; ++i) {
        triton_server::TritonServer::Instance().infer(model_name, model_version, inputs, outputs);
    }

    auto *n = outputs->tensors["output0"]->cpu<float>();
    for (int i = 0; i < 10; ++i) {
        InfoL << n[i] << std::endl;
    }

    TRITON_SERVER_UNINIT();
    return 0;
}
