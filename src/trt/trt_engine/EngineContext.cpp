//
// Created by lijin on 2023/12/21.
//

#include "EngineContext.h"

namespace TRT {
void EngineContext::destroy() {
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
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