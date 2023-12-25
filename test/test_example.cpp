#include <opencv2/opencv.hpp>

#include "infer/MultipleInferenceInstances.h"
#include "trt/yolo/Yolov5DetectPipeline.h"
#include "trt/yolo/Yolov5DetectionInfer.h"

int main() {
    std::string input_stream_url =
        "rtsp://192.168.161.149:1554/gate/f0389185-0fc2-49cf-d13d-38342b00fd95";
    std::string output_stream_url = "rtmp://192.168.161.149/yolov5/test";

    std::string model_path = "/root/trt_projects/infer-main/workspace/yolov5m.fp32.16bacth.engine";
    std::string label_path = "/tmp/tmp.wz9qvcR2y8/resource/labels/coco.labels";
    int         max_batch_size = 16;
    std::vector<int> device_list{1, 0, 0, 0};

    auto trt_instance =
        std::make_shared<infer::MultipleInferenceInstances<infer::Yolov5DetectionInfer>>(
            "yolov5", device_list, model_path, label_path);

    std::vector<pipeline::Yolov5DetectPipeline::ptr> pipelines;
    for (int i = 0; i < 3; i++) {
        auto pipeline = std::make_shared<pipeline::Yolov5DetectPipeline>(
            "test_pipeline_" + std::to_string(i), input_stream_url,
            output_stream_url + std::to_string(i), trt_instance);
        pipelines.push_back(pipeline);
    }
    for (auto &pipeline : pipelines) {
        pipeline->Start();
    }
    getchar();
}