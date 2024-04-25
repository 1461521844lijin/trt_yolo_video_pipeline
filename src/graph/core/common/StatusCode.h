//
// Created by lijin on 2023/12/18.
//

#ifndef VIDEOPIPELINE_STATUSCODE_H
#define VIDEOPIPELINE_STATUSCODE_H

namespace GraphCore {

enum StatusCode {
    OK = 0,  // 正常

    // 节点状态
    NodeBufferOver,  // 节点缓冲溢出
    NodeTimeout,     // 节点处理超时
    NodeError,       // 节点处理错误
    NodeExit,        // 节点退出
    NodeStop,        // 节点停止

    // ffmpeg 状态
    FFMpegReadError,  // ffmpeg 读取错误
    FFMpegWriteError, // ffmpeg 写入错误
    FFmpegDecodeError, // ffmpeg 解码错误
    FFmpegEncodeError, // ffmpeg 编码错误

    // 模型状态

    // 管道状态
    PipeLineError,  // 管道错误

};

}

#endif  // VIDEOPIPELINE_STATUSCODE_H
