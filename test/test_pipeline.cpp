//
// Created by lijin on 2023/12/6.
//

#include "trt/cpm.hpp"
#include "trt/infer.hpp"
#include "trt/yolo.hpp"

#include "pipeline/Yolov8Pipeline.h"

#include <any>
#include <iostream>

int main() {
    std::string model_path     = "****";
    std::string model_path_seg = "*******";
    int         max_batch_size = 16;
    std::string stream_url     = "rtmp://**";
    std::string output_url     = "rtmp://**";

    // 并发推流的数量
    // 本地测试时用的是从流媒体服务器分发的视频流
  // 因此输入流是相同视频流拉取出来的一个不同连接，如果是相机最多拉取4~6路
  // 输出流通过后缀下标区分
  int push_stream_nums = 6;
  // 指定gpu运行id
  int device_id = 0;

  std::shared_ptr<cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>>
      trt_instance;
  trt_instance = std::make_shared<
      cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>>();
  trt_instance->start(
      [&] {
        return yolo::load(model_path, yolo::Type::V8, 0.25, 0.5, device_id);
      },
      max_batch_size);

  std::vector<pipeline::Yolov8Pipeline::ptr> pipelines;
  for (int i = 0; i < push_stream_nums; i++) {
    auto pipeline = std::make_shared<pipeline::Yolov8Pipeline>(
        "test_pipeline_" + std::to_string(i), stream_url,
        output_url + std::to_string(i), trt_instance);
    pipelines.push_back(pipeline);
  }
  for (auto &item : pipelines) {
    item->Start();
  }
  getchar();

  return 0;
}