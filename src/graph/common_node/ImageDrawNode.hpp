
#pragma once

#include <utility>

#include "graph/core/common/DetectionBox.h"
#include "graph/core/node/ProcessNode.h"

namespace Node {

class ImageDrawNode : public GraphCore::Node {
public:
    explicit ImageDrawNode(std::string name)
        : Node(std::move(name), GraphCore::NODE_TYPE::MID_NODE) {}

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
};

class ImageDrawTrackNode : public GraphCore::Node {
public:
    explicit ImageDrawTrackNode(std::string name)
        : Node(std::move(name), GraphCore::NODE_TYPE::MID_NODE) {}

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
};

class ImageDrawSegNode : public GraphCore::Node {
public:
    explicit ImageDrawSegNode(std::string name)
        : Node(std::move(name), GraphCore::NODE_TYPE::MID_NODE) {}

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
};

}  // namespace Node
