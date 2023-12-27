//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_CUDATIMER_H
#define VIDEOPIPELINE_CUDATIMER_H

namespace CUDATools {

// 辅助计算cuda kernel的运行时间
class CudaTimer {
public:
    CudaTimer();
    virtual ~CudaTimer();
    void  start(void *stream = nullptr);
    float stop(const char *prefix = "Timer", bool print = true);

private:
    void *start_, *stop_;
    void *stream_;
};

}  // namespace CUDATools

#endif  // VIDEOPIPELINE_CUDATIMER_H
