//
// Created by lijin on 2023/12/18.
//

#ifndef TRT_YOLOV8_IEVENTCALLBACK_H
#define TRT_YOLOV8_IEVENTCALLBACK_H

#include <functional>
#include <iostream>

namespace GraphCore {

/**
 *  通用的回调函数
 *  (tag, code, msg)-> code
 */
typedef std::function<int(std::string, int, std::string)> CallBackFunction;

#ifndef Debug
#    define default_cb                                                                             \
        [](const std::string &tag, int code, const std::string &msg) {                             \
            std::cout << "[" << tag << "]"                                                         \
                      << " code: " << code << " msg: " << msg << std::endl;                        \
            return 0;                                                                              \
        };
#else
#    define default_cb [](const std::string &tag, int code, const std::string &msg) {};
#endif

/**
 *  定义了一些回调接口基类
 */
class IEventCallBack {
protected:
    // 错误回调
    CallBackFunction error_cb = default_cb;
    // 处理超时回调
    CallBackFunction timeout_cb = default_cb;
    // 缓冲溢出回调
    CallBackFunction buffer_over_cb = default_cb;
    // 节点启动回调
    CallBackFunction before_start_cb = default_cb;
    CallBackFunction after_start_cb  = default_cb;
    // 节点退出回调
    CallBackFunction exit_cb = default_cb;

public:
    void set_error_cb(CallBackFunction event_cb) {
        error_cb = std::move(event_cb);
    }

    void set_timeout_cb(CallBackFunction event_cb) {
        timeout_cb = std::move(event_cb);
    }

    void set_buffer_over_cb(CallBackFunction event_cb) {
        buffer_over_cb = std::move(event_cb);
    }

    void set_exit_cb(CallBackFunction event_cb) {
        exit_cb = std::move(event_cb);
    }

public:
    void set_after_start_cb(CallBackFunction event_cb) {
        after_start_cb = std::move(event_cb);
    }

    void set_before_start_cb(CallBackFunction event_cb) {
        before_start_cb = std::move(event_cb);
    }
};

}  // namespace GraphCore
#endif  // TRT_YOLOV8_IEVENTCALLBACK_H
