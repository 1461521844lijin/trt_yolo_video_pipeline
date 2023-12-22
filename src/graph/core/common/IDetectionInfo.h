//
// Created by lijin on 2023/12/19.
//

#ifndef VIDEOPIPELINE_IDETECTIONINFO_H
#define VIDEOPIPELINE_IDETECTIONINFO_H

#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// 检测器通用属性接口
class IDetectInfo {
public:
    IDetectInfo()          = default;
    virtual ~IDetectInfo() = default;

public:
    /**
     * 获取最佳类别id
     * @param src            输入地址
     * @param classNums      类别数
     * @param bestClassScore 最大类别得分
     * @param bestClassId    最大类别id
     */
    static void get_best_class_id(const float *src,
                                  const int    classNums,
                                  float       &bestClassScore,
                                  int         &bestClassId) {
        bestClassScore = 0;
        bestClassId    = 0;
        for (int i = 0; i < classNums; ++i) {
            if (src[i] > bestClassScore) {
                bestClassScore = src[i];
                bestClassId    = i;
            }
        }
    }

    /**
     * 缩放坐标 将模型输出的坐标缩放到原始图像尺寸
     * @param imageShape         模型输入尺寸
     * @param coords             坐标
     * @param imageOriginalShape 原始图像尺寸
     */
    static void scaleCoords(const cv::Size &imageShape,
                            cv::Rect       &coords,
                            const cv::Size &imageOriginalShape) {
        float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                              (float)imageShape.width / (float)imageOriginalShape.width);

        int pad[2] = {
            (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
            (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

        coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
        coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

        coords.width  = (int)std::round(((float)coords.width / gain));
        coords.height = (int)std::round(((float)coords.height / gain));
    }

    /**
     * 加载类别名称, 一行一个类别名称
     * @param class_name_path
     */
    void load_class_names(const std::string &class_name_path) {
        std::vector<std::string> classNames;
        std::ifstream            infile(class_name_path);
        if (infile.good()) {
            std::string line;
            while (getline(infile, line)) {
                if (line.back() == '\r')
                    line.pop_back();
                classNames.emplace_back(line);
            }
            infile.close();
            set_class_names(classNames);
            set_class_nums(classNames.size());
        } else {
            throw std::runtime_error("Failed to access class name path: " + class_name_path);
        }
    }

    void set_class_nums(int class_nums) {
        m_class_nums = class_nums;
    }

    int get_class_nums() const {
        return m_class_nums;
    }

    void set_class_names(const std::vector<std::string> &class_names) {
        m_class_names = class_names;
    }

    std::vector<std::string> get_class_names() const {
        return m_class_names;
    }

    void set_confidence_threshold(float confidence_threshold) {
        m_confidence_threshold = confidence_threshold;
    }

    float get_confidence_threshold() const {
        return m_confidence_threshold;
    }

    void set_nms_threshold(float nms_threshold) {
        m_nms_threshold = nms_threshold;
    }

    float get_nms_threshold() const {
        return m_nms_threshold;
    }

protected:
    int                      m_class_nums;            // 检测器类别数
    std::vector<std::string> m_class_names;           // 检测器类别名称
    float                    m_confidence_threshold;  // 置信度阈值
    float                    m_nms_threshold;         // nms阈值
};

#endif  // VIDEOPIPELINE_IDETECTIONINFO_H
