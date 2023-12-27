//
// Created by lijin on 2023/12/21.
//

#include "CudaTimer.h"
#include "cuda_tools.h"
#include <cuda_runtime.h>

namespace CUDATools {

#define checkRuntime(call)                                                                         \
    do {                                                                                           \
        auto ___call__ret_code__ = (call);                                                         \
        if (___call__ret_code__ != cudaSuccess) {                                                  \
            INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                          \
                 cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__),   \
                 ___call__ret_code__);                                                             \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

CudaTimer::CudaTimer() {
    checkRuntime(cudaEventCreate((cudaEvent_t *)&start_));
    checkRuntime(cudaEventCreate((cudaEvent_t *)&stop_));
}

CudaTimer::~CudaTimer() {
    checkRuntime(cudaEventDestroy((cudaEvent_t)start_));
    checkRuntime(cudaEventDestroy((cudaEvent_t)stop_));
}

void CudaTimer::start(void *stream) {
    stream_ = stream;
    checkRuntime(cudaEventRecord((cudaEvent_t)start_, (cudaStream_t)stream_));
}

float CudaTimer::stop(const char *prefix, bool print) {
    checkRuntime(cudaEventRecord((cudaEvent_t)stop_, (cudaStream_t)stream_));
    checkRuntime(cudaEventSynchronize((cudaEvent_t)stop_));

    float latency = 0;
    checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t)start_, (cudaEvent_t)stop_));

    if (print) {
        printf("[%s]: %.5f ms\n", prefix, latency);
    }
    return latency;
}

}  // namespace CUDATools