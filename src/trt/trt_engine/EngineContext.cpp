//
// Created by lijin on 2023/12/21.
//

#include "EngineContext.h"
#include <stdarg.h>

namespace TRT {
void EngineContext::destroy() {
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
}

static std::string file_name(const std::string &path, bool include_suffix) {
    if (path.empty())
        return "";

    int p = path.rfind('/');
    int e = path.rfind('\\');
    p     = std::max(p, e);
    p += 1;

    // include suffix
    if (include_suffix)
        return path.substr(p);

    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p)
        u = path.size();
    return path.substr(p, u - p);
}

void __log_func(const char *file, int line, const char *fmt, ...) {
    va_list vl;
    va_start(vl, fmt);
    char        buffer[2048];
    std::string filename = file_name(file, true);
    int         n        = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
    vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
    fprintf(stdout, "%s\n", buffer);
}

bool EngineContext::construct(const void *pdata, size_t size) {
    destroy();

    if (pdata == nullptr || size == 0)
        return false;

    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger),
                                                    destroy_nvidia_pointer<nvinfer1::IRuntime>);
    if (m_runtime == nullptr)
        return false;

    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        m_runtime->deserializeCudaEngine(pdata, size, nullptr),
        destroy_nvidia_pointer<nvinfer1::ICudaEngine>);
    if (m_engine == nullptr)
        return false;

    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(
        m_engine->createExecutionContext(), destroy_nvidia_pointer<nvinfer1::IExecutionContext>);
    return m_context != nullptr;
}

EngineContext::~EngineContext() {
    destroy();
}
}  // namespace TRT