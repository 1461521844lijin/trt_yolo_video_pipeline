//
// Created by jin_l on 2024/4/15.
//

#include "infer/MultipleInferenceInstances.h"
#include "trt/yolo/YoloDetectPipeline.h"
#include "trt/yolo/YoloDetectionInfer.h"

int main() {
    std::string input_stream_url  = "输入流路径";
    std::string output_stream_url = "输出流路径";
    std::string model_path        = "TRTengine模型文件路径";
    std::string label_path        = "检测分类类别文件路径";
    int         max_batch_size    = 16;    // 最大batch数
    float       config_threshold  = 0.25;  // 检测阈值
    float       nms_threshold     = 0.5;   // nms阈值

    // 模型实例数量列表，列表为模型实例数，每个元素代表该模型实例在哪张显卡上的下标
    std::vector<int> device_list{0, 0, 1, 1};
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

    // 创建处理pipeline
    auto pipeline = std::make_shared<pipeline::YoloDetectPipeline>(
        "test_pipeline", input_stream_url, output_stream_url, trt_instance);

    // 启动流水线
    pipeline->Start();

    record::RecordConfig record_config;
    record_config.record_type = record::RecordType::VIDEO_RECORD;  // 录制类型 mp4视频
    record_config.file_name   = "test.record.mp4";                 // 录制文件名
    record_config.save_path   = "./";                              // 录制文件保存路径
    record_config.duration    = 60;                                // 录制时长 60s
    //这里的高度和宽度是源视频的高度和宽度，需要保持一致
    record_config.src_height = 1080;  // 源视频高度
    record_config.src_width  = 1920;  // 源视频宽度
    record_config.dst_height = 1080;  // 目标视频高度
    record_config.dst_width  = 1920;  // 目标视频宽度

    // 向流水线中添加录制任务
    pipeline->add_record_task(std::make_shared<Data::Mp4RecordControlData>(record_config));

    getchar();
}