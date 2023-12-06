//
// Created by lijin on 2023/12/6.
//

#include "Yolov8Pipeline.h"
#include "ffmpeg/FFmpegInputNode.h"
#include "ffmpeg/StreamPusher.h"
#include "trt/image_draw_node.hpp"
#include "trt/trt_node.hpp"

namespace pipeline {
Yolov8Pipeline::Yolov8Pipeline(std::string task_name, std::string input_url,
                               std::string output_url,
                               const TRTInstancePtr &trt_instance,
                               int output_width, int output_height,
                               int output_fps, int output_bitrate)
    : Pipeline(std::move(task_name)), m_input_url(std::move(input_url)),
      m_output_url(std::move(output_url)), m_output_width(output_width),
      m_output_height(output_height), m_output_fps(output_fps),
      m_output_bitrate(output_bitrate), trt_instance(trt_instance) {}
bool Yolov8Pipeline::Init() {
  if (m_initialized) {
    return true;
  }

  // todo 在初始化节点时有一些参数校验工作没有实现，如果输入的参数不正确会导致bu
  auto ffmpeg_input_node =
      FFmpeg::create_ffmpeg("nput_node_" + get_name(), m_input_url);
  auto trt_node = std::make_shared<trt::TrtNode>("trt_node_" + get_name());
  auto trt_draw_node =
      std::make_shared<trt::ImageDrawNode>("trt_draw_node_" + get_name());
  auto ffmpeg_output_node =
      std::make_shared<FFmpeg::StreamPusher>("output_node_" + get_name());
  trt_node->set_trt_instance(trt_instance);

  Base::LinkNode(ffmpeg_input_node, trt_node);
  Base::LinkNode(trt_node, trt_draw_node);
  Base::LinkNode(trt_draw_node, ffmpeg_output_node);

  ffmpeg_output_node->set_frommat(1920, 1080, AV_PIX_FMT_BGR24);
  ffmpeg_output_node->set_tomat(1920, 1080, AV_PIX_FMT_YUV420P);
  ffmpeg_output_node->Open(m_output_url, false);

  m_nodes.push_back(ffmpeg_input_node);
  m_nodes.push_back(trt_node);
  m_nodes.push_back(trt_draw_node);
  m_nodes.push_back(ffmpeg_output_node);

  m_initialized = true;
  return true;
}
} // namespace pipeline