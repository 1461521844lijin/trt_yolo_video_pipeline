//
// Created by Lenovo on 24-7-25.
//


#include "infer/MultipleInferenceInstances.h"
#include "triton/triton_infer/YoloDetectPipeline.h"
#include "triton/triton_infer/YoloDetectionInfer.h"

int main() {

//    auto default_channel = std::make_shared<toolkit::ConsoleChannel>("default",  toolkit::LError);
//    toolkit::Logger::Instance().add(default_channel);
//    toolkit::Logger::Instance().setWriter(std::make_shared<toolkit::AsyncLogWriter>());

    InfoL << "test triton yolo pipeline";

    // 流水线数量，即视频流数量
    // ps：这里只是示例，实际应用中可以根据实际情况修改，输入的视频都是同一个视频流
    int         stream_num        = 1;
    std::string input_stream_url  = "rtmp://";
    std::string output_stream_url = "rtmp://";
    std::string model_path        = "yolov8_trt";
    std::string label_path        = "../../resource/labels/coco.labels";
    std::string triton_client_uri = ""; // triton grpc uri
    int         max_batch_size    = 4;    // 最大batch数
    float       config_threshold  = 0.25;  // 检测阈值
    float       nms_threshold     = 0.5;   // nms阈值

    auto             type = infer::YoloType::V8;  // 模型类型

    // 创建多卡多实例推理对象
    auto trt_instance =
        std::make_shared<infer::YoloDetectionInfer>(
            "triton_instance", 0, triton_client_uri, model_path, label_path, type, config_threshold,
            nms_threshold, max_batch_size);
    if (!trt_instance->init()) {
        std::cout << "init failed" << std::endl;
        return -1;
    }

    // 添加多个流水线都用同一个多实例推理推理对象
    std::shared_ptr<pipeline::YoloDetectPipeline> pipeline_list[stream_num];
    for(int i = 0; i < stream_num; i++){
        // 创建处理pipeline
        auto pipeline = std::make_shared<pipeline::YoloDetectPipeline>(
            "test_pipeline_"+std::to_string(i),
            input_stream_url,
            output_stream_url+std::to_string(i),
            trt_instance,
            1280, 720
        );
        // 启动流水线
        pipeline->Start();
        pipeline_list[i] = pipeline;
    }

    // 阻塞主线程
    getchar();
}