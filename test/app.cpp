#include <opencv2/opencv.hpp>

#include "trt/cpm.hpp"
#include "trt/infer.hpp"
#include "trt/yolo.hpp"

#include "ffmpeg/FFmpegInputNode.h"
#include "ffmpeg/StreamPusher.h"
#include "trt/trt_node.hpp"
#include "trt/image_draw_node.hpp"

int main()
{

    std::string model_path = "/root/trt_projects/infer-main/workspace/yolov8n.transd.engine";
    std::string model_path_seg = "/root/trt_projects/infer-main/workspace/yolov8n-seg.b1.transd.engine";
    int max_batch_size = 16;
    std::string stream_url = "rtsp://*********";
    std::string output_url = "rtmp://*********";

    std::shared_ptr<cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>> trt_instance;
    trt_instance = std::make_shared<cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>>();
    trt_instance->start([&]
                        { return yolo::load(model_path_seg, yolo::Type::V8Seg); },
                        max_batch_size);

    auto ffmpeg_input_node = FFmpeg::create_ffmpeg("ffmpeg_input_node", stream_url);
    auto trt_node = std::make_shared<trt::TrtNode>("trt_node");
    auto trt_draw_node = std::make_shared<trt::ImageDrawSegNode>("trt_draw_node");
    auto ffmpeg_output_node = std::make_shared<FFmpeg::StreamPusher>("ffmpeg_output_node");
    trt_node->set_trt_instance(trt_instance);

    Base::LinkNode(ffmpeg_input_node, trt_node);
    Base::LinkNode(trt_node, trt_draw_node);
    Base::LinkNode(trt_draw_node, ffmpeg_output_node);

    ffmpeg_output_node->set_frommat(2560, 1440, AV_PIX_FMT_BGR24);
    ffmpeg_output_node->set_tomat(1920, 1080, AV_PIX_FMT_YUV420P);
    ffmpeg_output_node->Open(output_url, false);

    trt_draw_node->Start();
    ffmpeg_output_node->Start();
    trt_node->Start();
    ffmpeg_input_node->Start();

    getchar();
}