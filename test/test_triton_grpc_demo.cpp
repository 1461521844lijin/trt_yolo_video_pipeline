//
// Created by Lenovo on 24-7-12.
//

#include "triton/triton_client/TritonGrpcClient.h"

int main() {
    // 1、初始化客户端链接
    triton_client::TritonGrpcClient::ptr client =
        std::make_shared<triton_client::TritonGrpcClient>();
    client->init_client("192.168.161.152:11001", false);

    // 2、初始化推理输出输出对象
    client->init_model_infer("yolov8_fp16_trt", triton_client::DateTransMode::CUDASHM);

    // 3、构建输入数据
    CUDA::Tensor::ptr input = std::make_shared<CUDA::Tensor>();
    input->resize({1, 3, 640, 640});
    input->set_to(1);

    // 4、设置输入数据
    client->AddInput("images", input);

    // 5、执行推理
    for (int i = 0; i < 100; ++i) {
        client->infer();
    }

    // 6、获取输出数据
    CUDA::Tensor::ptr output;
    client->GetOutput("output0", output);
    InfoL << output->descriptor();

    return 0;
}