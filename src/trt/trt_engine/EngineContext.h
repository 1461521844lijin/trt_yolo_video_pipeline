//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_ENGINEH
#define VIDEOPIPELINE_ENGINEH

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_kernels/cuda_tools/ilogger.hpp>
#include <memory>

namespace TRT {

#define INFO(...) __log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char *file, int line, const char *fmt, ...);
class __native_nvinfer_logger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, const char *msg) noexcept override {
        if (severity == Severity::kINTERNAL_ERROR) {
            INFO("NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        } else if (severity == Severity::kERROR) {
            INFO("NVInfer: %s", msg);
        } else if (severity == Severity::kWARNING) {
            INFO("NVInfer: %s", msg);
        } else if (severity == Severity::kINFO) {
            INFO("NVInfer: %s", msg);
        } else {
            INFO("%s", msg);
        }
    }
};
static __native_nvinfer_logger gLogger;

template <typename _T>
static void destroy_nvidia_pointer(_T *ptr) {
    if (ptr)
        ptr->destroy();
}

class EngineContext {
public:
    virtual ~EngineContext();

    bool construct(const void *pdata, size_t size);

private:
    void destroy();

public:
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;
    std::shared_ptr<nvinfer1::ICudaEngine>       m_engine;
    std::shared_ptr<nvinfer1::IRuntime>          m_runtime = nullptr;
};

}  // namespace TRT
#endif  // VIDEOPIPELINE_ENGINEm_contextH
