#pragma once

#include "base/ProcessNode.hpp"

#include "trt/yolo.hpp"

namespace trt
{

class ImageDrawNode : public Base::Node
{
public:
    ImageDrawNode(std::string name) : Node(name) {}
private:
    void worker() override;
};


class ImageDrawSegNode : public Base::Node
{
public:
    ImageDrawSegNode(std::string name) : Node(name) {}
private:
    void worker() override;

};


}  // namespace trt