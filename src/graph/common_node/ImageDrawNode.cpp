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
    auto image = data->Get<MAT_IMAGE_TYPE>(MAT_IMAGE);
    //    auto box_array = data->Get<DETECTBOX_FUTURE_TYPE>(DETECTBOX_FUTURE).get();

    // 图像最多等待120ms
    auto status =
        data->Get<DETECTBOX_FUTURE_TYPE>(DETECTBOX_FUTURE).wait_for(std::chrono::milliseconds(120));
    if (status == std::future_status::timeout) {
        printf("ImageDrawNode: %s wait for future timeout\n", getName().c_str());
        return nullptr;
    }
    auto box_array = data->Get<DETECTBOX_FUTURE_TYPE>(DETECTBOX_FUTURE).get();

    for (auto &obj : box_array) {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                      cv::Scalar(b, g, r), 2);
        cv::putText(image, obj.class_name, cv::Point(obj.left, obj.top - 5),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(b, g, r), 1);

        //        printf("class_name: %s, confidence: %f, left: %f, top: %f, right: %f, bottom:
        //        %f\n",
        //               obj.class_name.c_str(), obj.confidence, obj.left, obj.top, obj.right,
        //               obj.bottom);

        if (obj.mask.empty())
            continue;

        // 判断是否越界
        if (obj.left < 0 || obj.right < 0 || obj.top < 0 || obj.bottom < 0)
            continue;
        if (obj.left > image.cols || obj.right > image.cols || obj.top > image.rows ||
            obj.bottom > image.rows)
            continue;
        // 转为三通道图像并赋予随机颜色
        cv::cvtColor(obj.mask, obj.mask, cv::COLOR_GRAY2BGR);
        // 阈值化
        cv::threshold(obj.mask, obj.mask, 100, 255, cv::THRESH_BINARY);
        cv::Vec3b color = cv::Vec3b(b, g, r);
        // 将掩码中的255像素赋予随机颜色
        obj.mask.setTo(color, obj.mask == cv::Vec3b(255, 255, 255));
        // 将掩码缩放到原图像大小
        cv::Mat resize_mask;
        cv::resize(obj.mask, resize_mask, cv::Size(obj.right - obj.left, obj.bottom - obj.top));
        // 获取原始图像的ROI
        cv::Mat imageROI =
            image(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top));
        //        cv::imwrite("mask.jpg", obj.mask);
        // 将掩码叠加到ROI上
        cv::addWeighted(imageROI, 1, resize_mask, 0.8, 1, imageROI);
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
