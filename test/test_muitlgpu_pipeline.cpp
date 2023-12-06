//
// Created by lijin on 2023/12/6.
//

//
// Created by lijin on 2023/12/6.
//

#include "pipeline/MultiGPUYolov8TRTInfer.h"
#include "pipeline/Yolov8Pipeline.h"

int main() {

  std::string model_path = "****";
  std::string model_path_seg = "*******";
  int max_batch_size = 16;
  std::string stream_url = "rtmp://**";
  std::string output_url = "rtmp://**";

  // 并发数量
  int push_stream_nums = 10;

  // 在gpu 0和1上各创建2各实例
  std::vector<int> device_id_list = {0, 1, 0, 1};
  auto infer_instance = std::make_shared<infer::MultiGPUYolov8TRTInfer>(
      "test_infer", model_path, max_batch_size, device_id_list, yolo::Type::V8);

  std::vector<pipeline::Yolov8Pipeline::ptr> pipelines;
  for (int i = 0; i < push_stream_nums; i++) {
    auto pipeline = std::make_shared<pipeline::Yolov8Pipeline>(
        "test_pipeline_" + std::to_string(i), stream_url,
        output_url + std::to_string(i), infer_instance);
    pipelines.push_back(pipeline);
  }
  for (auto &item : pipelines) {
    item->Start();
  }
  getchar();

  return 0;
}