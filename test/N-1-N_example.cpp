//
// Created by jin_l on 2024/4/15.
//

#include "infer/MultipleInferenceInstances.h"
#include "trt/yolo/YoloDetectPipeline.h"
#include "trt/yolo/YoloDetectionInfer.h"

int main() {

    /*
     * 该示例展示了如何创建多个流水线，每个流水线都使用同一个多实例推理推理对象
     * 示例中创建了5个流水线（5路视频），分别在1号和2号显卡上各创建两个模型实例，进行轮询负载推理
     */

    // 流水线数量，即视频流数量 ps：这里只是示例，实际应用中可以根据实际情况修改，输入的视频都是同一个视频流
    int stream_num = 20;
    std::string input_stream_url  = "rtmp://192.168.161.149:11935/gate/f9edf7cb-b05a-407f-9cc1-aab7494b7cbf";
    std::string output_stream_url = "rtmp://192.168.161.149/test/yolov8n";
    std::string model_path        = "/root/trt_projects/yolov8n.transd.engine";
    std::string label_path        = "/tmp/tmp.C5wIfz4ahJ/resource/labels/coco.labels";
    int         max_batch_size    = 16;    // 最大batch数
    float       config_threshold  = 0.25;  // 检测阈值
    float       nms_threshold     = 0.5;   // nms阈值

    // 模型实例数量列表，列表为模型实例数，每个元素代表该模型实例在哪张显卡上的下标
    // 该示例中，1号和2号显卡上各创建两个模型实例
    std::vector<int> device_list{0,0,0};
    auto             type = infer::YoloType::V8;  // 模型类型

    // 创建多卡多实例推理对象
    auto trt_instance =
        std::make_shared<infer::MultipleInferenceInstances<infer::YoloDetectionInfer>>(
            "trt_instance", device_list, model_path, label_path, type, config_threshold,
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
            trt_instance
            );
        // 启动流水线
        pipeline->Start();
        pipeline_list[i] = pipeline;
    }

    // 阻塞主线程
    getchar();
}