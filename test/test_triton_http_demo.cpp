//
// Created by Lenovo on 24-7-12.
//

#include "triton/triton_client/TritonHttpClient.h"



int main() {
    triton_client::TritonHttpClient::ptr client = std::make_shared<triton_client::TritonHttpClient>();
    client->init_client("192.168.161.152:11000", false);
    std::string repository_index = client->get_repository_index();
    std::cout << repository_index << std::endl;

    std::cout<< "model config: " << client->get_model_config("yolov8n_batch_onnx") << std::endl;

    std::cout<< "model metadata: " << client->get_model_metadata("yolov8n_batch_onnx") << std::endl;

    client->unload_model("yolov8n_onnx");
    std::cout <<"unload: " <<client->get_repository_index() << std::endl;

//    client->load_model("yolov8n_onnx");
    std::cout <<"reload: " <<client->get_repository_index() << std::endl;

    return 0;
}