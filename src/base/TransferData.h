#pragma once
#include <opencv2/core/core.hpp>
#include <memory>
#include <string>
#include <future>
#include "trt/yolo.hpp"

namespace Data {

class Input_Data {
public:
    virtual ~Input_Data() {}

};



class Decode_Data : public Input_Data {
public:
    Decode_Data() = default;

    Decode_Data(int ow, int oh, int sw, int sh)
        : original_w(ow),
          original_h(oh),
          scale_w(sw),
          scale_h(sh)

    {
        original_image = cv::Mat(original_h, original_w, CV_8UC3);
        scale_image    = cv::Mat(scale_h, scale_w, CV_8UC3);
    }

    cv::Mat     original_image;
    cv::Mat     scale_image;
    int         original_w, original_h, scale_w, scale_h;
    std::string stream_id;

    std::shared_future<yolo::BoxArray> boxarray_fu;
};


}  // namespace Data