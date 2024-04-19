
#pragma once

#include "graph/core/common/DetectionBox.h"
#include "graph/core/node/ProcessNode.h"
#include "utils/httplib.h"
#include "utils/logger.h"
#include <utility>

namespace Node {

/*
 推送json格式  这部分根据实际情况修改就好

    {
        "img_base64": "图片base64数据",
        img_width: 1920,
        img_height: 1080,
        "frame_id": 1,
        "post_time": 123456789,
        "detection_boxes": [
            {,
                "class_label": "person",
                "confidence": 0.9,
                “left”: 0.1,
                “top”: 0.1,
                right: 0.2,
                bottom: 0.2
            },
            {
                "class_label": "person",
                "confidence": 0.9,
                “left”: 0.1,
                “top”: 0.1,
                right: 0.2,
                bottom: 0.2
            }
        ]
    }

*/

/**
 * 用于将检测帧目标框数据上传到上级服务器
 * 使用场景：检测目标帧数据上传，例如边缘设备、或者不需要推流的场景
 */
class KeyframePostNode : public GraphCore::Node {
public:
    /**
     * @param name  节点名称
     * @param client_url  上传的服务器地址
     * @param post_cycle  上传周期 单位ms 默认0即每一帧都上传
     */
    explicit KeyframePostNode(const std::string &name,
                              const std::string &client_url,
                              const std::string &post_url,
                              int64              post_cycle = 0);

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;

private:
    std::string               m_client_url;          // 上传的服务器地址  http://ip:port
    std::string               m_post_url;            // 请求接口后缀  /api/v1/xxx
    httplib::Client           m_client;              // http client
    int64                     m_post_cycle     = 0;  // 上传周期 单位s
    int64                     m_last_post_time = 0 ;  // 上次上传时间
};
}  // namespace Node
