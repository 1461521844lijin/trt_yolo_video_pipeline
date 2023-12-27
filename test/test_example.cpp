#include "ffmpeg/record/Mp4RecordControlData.h"
#include "infer/MultipleInferenceInstances.h"
#include "trt/yolo/YoloDetectPipeline.h"
#include "trt/yolo/YoloDetectionInfer.h"

int main() {
    std::string input_stream_url =
        "rtmp://192.168.161.149:11935/gate/f0389185-0fc2-49cf-d13d-38342b00fd95";
    std::string input_local_file  = "/tmp/tmp.wz9qvcR2y8/resource/test_video/car_test.mp4";
    std::string output_stream_url = "rtmp://192.168.161.149/yolov5/test";

    std::string model_path = "/root/trt_projects/infer-main/workspace/yolov5m.fp32.16bacth.engine";
    std::string v8_model_path =
        "/root/trt_projects/infer-main/workspace/yolov8n.32batch.fp16.engine";
    std::string v8seg_model_path =
        "/root/trt_projects/infer-main/workspace/yolov8n-seg.b1.transd.engine";
    std::string      label_path     = "/tmp/tmp.wz9qvcR2y8/resource/labels/coco.labels";
    int              max_batch_size = 16;
    std::vector<int> device_list{0};
    auto             type = infer::YoloType::V8;

    auto trt_instance =
        std::make_shared<infer::MultipleInferenceInstances<infer::YoloDetectionInfer>>(
            "yolov5", device_list, v8_model_path, label_path, type);
    if (!trt_instance->init()) {
        std::cout << "init failed" << std::endl;
        return -1;
    }
    std::vector<pipeline::YoloDetectPipeline::ptr> pipelines;
    for (int i = 0; i < 1; i++) {
        auto pipeline = std::make_shared<pipeline::YoloDetectPipeline>(
            "test_pipeline_" + std::to_string(i), input_stream_url,
            output_stream_url + std::to_string(i), trt_instance);
        pipelines.push_back(pipeline);
    }
    for (auto &pipeline : pipelines) {
        pipeline->Start();
    }

    //    getchar();

    // 录制一个20s的视频
    //    record::RecordConfig config;
    //    config.save_path   = "/tmp/tmp.wz9qvcR2y8/resource/test_video/";
    //    config.src_width   = 1280;
    //    config.src_height  = 720;
    //    config.dst_width   = 1280;
    //    config.dst_height  = 720;
    //    config.file_name   = "test2.mp4";
    //    config.record_type = record::RecordType::VIDEO_RECORD;
    //    config.duration    = 20;
    //    Data::Mp4RecordControlData::ptr control_data =
    //        std::make_shared<Data::Mp4RecordControlData>(config);
    //    pipelines[0]->add_record_task(control_data);
    //    std::cout << "start record" << std::endl;

    getchar();
}