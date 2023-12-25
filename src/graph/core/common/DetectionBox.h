//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_DETECTIONBOX_H
#define VIDEOPIPELINE_DETECTIONBOX_H

#include "opencv2/opencv.hpp"

struct DetectBox {
    float       left        = -1;
    float       top         = -1;
    float       right       = -1;
    float       bottom      = -1;
    float       confidence  = -1;
    int         class_label = -1;
    int         track_id    = -1;
    std::string class_name{};
    cv::Mat     mask{};

    DetectBox() = default;

    DetectBox(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          confidence(confidence),
          class_label(class_label) {}
};

typedef std::vector<DetectBox> DetectBoxArray;

#endif  // VIDEOPIPELINE_DETECTIONBOX_H
