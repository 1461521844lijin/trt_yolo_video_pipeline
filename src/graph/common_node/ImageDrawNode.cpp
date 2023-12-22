#include "ImageDrawNode.hpp"
#include <opencv2/opencv.hpp>
#include <tuple>
namespace Node {

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
    const int   h_i = static_cast<int>(h * 6);
    const float f   = h * 6 - h_i;
    const float p   = v * (1 - s);
    const float q   = v * (1 - f * s);
    const float t   = v * (1 - (1 - f) * s);
    float       r, g, b;
    switch (h_i) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
        default:
            r = 1;
            g = 1;
            b = 1;
            break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                           static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

Data::BaseData::ptr ImageDrawNode::handle_data(Data::BaseData::ptr data) {
    auto image     = data->Get<MAT_IMAGE_TYPE>(MAT_IMAGE);
    auto box_array = data->Get<DETECTBOX_FUTURE_TYPE>(DETECTBOX_FUTURE).get();
    for (auto &obj : box_array) {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      cv::Scalar(b, g, r), 2);
    }
    return data;
}

Data::BaseData::ptr ImageDrawSegNode::handle_data(Data::BaseData::ptr data) {
    // todo
    return data;
}
Data::BaseData::ptr ImageDrawTrackNode::handle_data(Data::BaseData::ptr data) {
    // todo
    return data;
}
}  // namespace Node
