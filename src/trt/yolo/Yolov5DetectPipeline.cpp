//
// Created by lijin on 2023/12/22.
//

#include "Yolov5DetectPipeline.h"
#include "ffmpeg/ffmpeg_node/FFmpegOutputNode.h"
#include "ffmpeg/ffmpeg_node/FFmpegReadNode.h"
#include "graph/common_node/ImageDrawNode.hpp"
#include "infer/InferNode.h"
namespace pipeline {

Yolov5DetectPipeline::Yolov5DetectPipeline(std::string              task_name,
                                           std::string              input_url,
                                           std::string              output_url,
                                           const infer::Infer::ptr &trt_instance,
                                           int                      output_width,
                                           int                      output_height,
                                           int                      output_fps,
                                           int                      output_bitrate)
    : Pipeline(std::move(task_name)),
      m_input_url(std::move(input_url)),
      m_output_url(std::move(output_url)),
      m_output_width(output_width),
      m_output_height(output_height),
      m_output_fps(output_fps),
      m_output_bitrate(output_bitrate),
      m_trt_instance(trt_instance) {}

bool Yolov5DetectPipeline::Init() {
    if (m_initialized) {
        return true;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_trt_instance) {
        std::cerr << "trt实例不存在" << std::endl;
        return false;
    }
    auto ffmpeg_input_node =
        std::make_shared<Node::FFmpegReadNode>("ffmpeg_input_node", m_input_url);
    ASSERT_INIT(ffmpeg_input_node->Init());
    auto trt_node      = std::make_shared<Node::InferNode>("trt_node");
    auto trt_draw_node = std::make_shared<Node::ImageDrawNode>("trt_draw_node");

    auto [input_w, input_h, fps, bitrate] = ffmpeg_input_node->get_video_info();
    auto ffmpeg_output_node =
        std::make_shared<Node::FFmpegOutputNode>("ffmpeg_output_node", m_output_url, input_w,
                                                 input_h, AV_PIX_FMT_BGR24, m_output_width,
                                                 m_output_height, AV_PIX_FMT_YUV420P, fps, bitrate);
    ASSERT_INIT(ffmpeg_output_node->Init());

    trt_node->set_trt_instance(m_trt_instance);

    GraphCore::LinkNode(ffmpeg_input_node, trt_node);
    GraphCore::LinkNode(trt_node, trt_draw_node);
    GraphCore::LinkNode(trt_draw_node, ffmpeg_output_node);

    m_nodes["ffmpeg_input_node"]  = ffmpeg_input_node;
    m_nodes["trt_node"]           = trt_node;
    m_nodes["trt_draw_node"]      = trt_draw_node;
    m_nodes["ffmpeg_output_node"] = ffmpeg_output_node;

    m_initialized = true;
    return true;
}
}  // namespace pipeline