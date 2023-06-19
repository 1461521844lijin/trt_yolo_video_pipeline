#include "image_draw_node.hpp"
#include <opencv2/opencv.hpp>
namespace trt
{

    static const char *cocolabels[] = {"person", "bicycle", "car",
                                       "motorcycle", "airplane", "bus",
                                       "train", "truck", "boat",
                                       "traffic light", "fire hydrant", "stop sign",
                                       "parking meter", "bench", "bird",
                                       "cat", "dog", "horse",
                                       "sheep", "cow", "elephant",
                                       "bear", "zebra", "giraffe",
                                       "backpack", "umbrella", "handbag",
                                       "tie", "suitcase", "frisbee",
                                       "skis", "snowboard", "sports ball",
                                       "kite", "baseball bat", "baseball glove",
                                       "skateboard", "surfboard", "tennis racket",
                                       "bottle", "wine glass", "cup",
                                       "fork", "knife", "spoon",
                                       "bowl", "banana", "apple",
                                       "sandwich", "orange", "broccoli",
                                       "carrot", "hot dog", "pizza",
                                       "donut", "cake", "chair",
                                       "couch", "potted plant", "bed",
                                       "dining table", "toilet", "tv",
                                       "laptop", "mouse", "remote",
                                       "keyboard", "cell phone", "microwave",
                                       "oven", "toaster", "sink",
                                       "refrigerator", "book", "clock",
                                       "vase", "scissors", "teddy bear",
                                       "hair drier", "toothbrush"};

    void ImageDrawNode::worker()
    {

        std::shared_ptr<Data::Input_Data> data;
        while (m_run)
        {
            for (auto &input : m_input_buffers)
            {
                if (input.second->Pop(data))
                {
                    std::shared_ptr<Data::Decode_Data> decode_data =
                        std::dynamic_pointer_cast<Data::Decode_Data>(data);

                    auto boxarray = decode_data->boxarray_fu.get();
                    for (auto &obj : boxarray)
                    {
                        uint8_t b, g, r;
                        std::tie(b, g, r) = yolo::random_color(obj.class_label);
                        cv::rectangle(decode_data->original_image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 2);
                        auto name = cocolabels[obj.class_label];
                        auto caption = cv::format("%s %.2f", name, obj.confidence);
                        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                        cv::rectangle(decode_data->original_image, cv::Point(obj.left - 3, obj.top - 33),
                                      cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                        cv::putText(decode_data->original_image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
                    }
                    for (auto &output : m_output_buffers)
                    {
                        output.second->Push(decode_data);
                    }
                }
                else
                {
                    if (!m_run)
                        break;
                    std::unique_lock<std::mutex> lk(m_base_mutex);
                    m_base_cond->wait(lk);
                }
            }
        }
    }

    void ImageDrawSegNode::worker()
    {

        std::shared_ptr<Data::Input_Data> data;
        while (m_run)
        {
            for (auto &input : m_input_buffers)
            {
                if (input.second->Pop(data))
                {
                    std::shared_ptr<Data::Decode_Data> decode_data =
                        std::dynamic_pointer_cast<Data::Decode_Data>(data);
                    auto boxarray = decode_data->boxarray_fu.get();
                    for (auto &obj : boxarray)
                    {
                        uint8_t b, g, r;
                        std::tie(b, g, r) = yolo::random_color(obj.class_label);
                        cv::rectangle(decode_data->original_image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 2);
                        auto name = cocolabels[obj.class_label];
                        auto caption = cv::format("%s %.2f", name, obj.confidence);
                        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                        cv::rectangle(decode_data->original_image, cv::Point(obj.left - 3, obj.top - 33),
                                      cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                        cv::putText(decode_data->original_image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
                        // 判断是否越界
                        if (obj.left < 0 || obj.right < 0 || obj.top < 0 || obj.bottom < 0)
                            continue;
                        if (obj.left > decode_data->original_image.cols || obj.right > decode_data->original_image.cols || obj.top > decode_data->original_image.rows || obj.bottom > decode_data->original_image.rows)
                            continue;
                        if (obj.seg)
                        {
                            cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
                            // 转为三通道图像并赋予随机颜色
                            cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
                            // 阈值化
                            cv::threshold(mask, mask, 50, 255, cv::THRESH_BINARY);
                            // 生成随机颜色
                            cv::Vec3b color = cv::Vec3b(b, g, r);
                            // 将掩码中的255像素赋予随机颜色
                            mask.setTo(color, mask > cv::Vec3b(155, 155, 155));
                            // 将掩码缩放到原图像大小
                            cv::Mat resize_mask;
                            cv::resize(mask, resize_mask, cv::Size(obj.right - obj.left, obj.bottom - obj.top));
                            // 获取原始图像的ROI
                            cv::Mat imageROI = decode_data->original_image(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top));
                            // 将掩码叠加到ROI上
                            cv::addWeighted(imageROI, 1, resize_mask, 0.8, 1, imageROI);
                        }
                    }
                    for (auto &output : m_output_buffers)
                    {
                        output.second->Push(decode_data);
                    }
                }
                else
                {
                    if (!m_run)
                        break;
                    std::unique_lock<std::mutex> lk(m_base_mutex);
                    m_base_cond->wait(lk);
                }
            }
        }
    }

}
