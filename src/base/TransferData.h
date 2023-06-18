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

    Decode_Data(int ow, int oh)
        : original_w(ow),
          original_h(oh)

    {
        original_image = cv::Mat(original_h, original_w, CV_8UC3);
    }

    cv::Mat     original_image;
    int         original_w, original_h;
    std::shared_future<yolo::BoxArray> boxarray_fu;
};


}  // namespace Data