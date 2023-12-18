//
// Created by lijin on 2023/12/18.
//

#ifndef VIDEOPIPELINE_STATUSCODE_H
#define VIDEOPIPELINE_STATUSCODE_H

namespace GraphCore {

enum StatusCode {
    OK = 0,         // 正常
    NodeBufferOver, // 节点缓冲溢出
    NodeTimeout,    // 节点处理超时
    NodeError,      // 节点处理错误
    NodeExit,       // 节点退出
    NodeStop,       // 节点停止
};

}

#endif  // VIDEOPIPELINE_STATUSCODE_H
