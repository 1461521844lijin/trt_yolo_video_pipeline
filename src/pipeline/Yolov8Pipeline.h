//
// Created by lijin on 2023/12/6.
//

#ifndef TRT_YOLOV8_YOLOV8PIPELINE_H
#define TRT_YOLOV8_YOLOV8PIPELINE_H

#include "Pipeline.h"
#include "trt/cpm.hpp"
#include "trt/infer.hpp"
#include "trt/yolo.hpp"

namespace pipeline {

class Yolov8Pipeline : public Pipeline {
public:
  typedef std::shared_ptr<Yolov8Pipeline> ptr;
  typedef std::shared_ptr<
      cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>>
      TRTInstancePtr;

private:
  TRTInstancePtr trt_instance;
  std::string m_input_url;  // rtsp url or video file path
  std::string m_output_url; // rtmp url
  int m_output_width = 1920;
  int m_output_height = 1080;
  int m_output_fps = 25;
  int m_output_bitrate = 1024 * 1024 * 2;

public:
  Yolov8Pipeline(std::string task_name, std::string input_url,
                 std::string output_url, const TRTInstancePtr &trt_instance,
                 int output_width = 1920, int output_height = 1080,
                 int output_fps = 25, int output_bitrate = 1024 * 1024 * 2);

private:
  bool Init() override;
};

} // namespace pipeline

#endif // TRT_YOLOV8_YOLOV8PIPELINE_H
