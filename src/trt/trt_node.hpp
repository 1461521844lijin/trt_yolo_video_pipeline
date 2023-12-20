#pragma once

#include "graph/core/node/ProcessNode.h"
#include "trt/cpm.hpp"
#include "trt/yolo.hpp"
namespace trt {

class TrtNode : public GraphCore::Node {
public:
    typedef std::shared_ptr<cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer>> TRTInstancePtr;

private:
    std::string    m_model_path;
    TRTInstancePtr m_trt_instance;

public:
    TrtNode(std::string name) : Node(name, GraphCore::NODE_TYPE::MID_NODE) {}
    void set_trt_instance(TRTInstancePtr &instance);

private:
    void worker() override;
};

}  // namespace trt