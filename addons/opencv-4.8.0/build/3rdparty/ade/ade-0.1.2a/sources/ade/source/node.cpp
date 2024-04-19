// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ade/node.hpp"

#include <memory>

#include "ade/util/assert.hpp"
#include "ade/util/algorithm.hpp"

#include "ade/graph.hpp"
#include "ade/edge.hpp"

namespace ade
{

Node::Node(Graph* parent):
    m_parent(parent)
{

}

Node::~Node()
{
    unlink();
}

void Node::unlink()
{
    ADE_ASSERT(nullptr != m_parent);
    for (auto& edge: m_inEdges) //TODO: join ranges
    {
        ADE_ASSERT(nullptr != edge);
        ADE_ASSERT(this == edge->m_nextNode);
        edge->m_nextNode = nullptr;
        m_parent->removeEdge(edge);
    }
    m_inEdges.clear();

    for (auto& edge: m_outEdges) //TODO: join ranges
    {
        ADE_ASSERT(nullptr != edge);
        ADE_ASSERT(this == edge->m_prevNode);
        edge->m_prevNode = nullptr;
        m_parent->removeEdge(edge);
    }
    m_outEdges.clear();
}

void Node::addInEdge(Edge* edge)
{
    ADE_ASSERT(nullptr != edge);
    ADE_ASSERT(m_inEdges.end() == util::find(m_inEdges, edge));
    m_inEdges.emplace_back(edge);
}

void Node::removeInEdge(Edge* edge)
{
    ADE_ASSERT(nullptr != edge);
    // Nodes usually have only a small amount of connections so linear search should be fine here
    auto it = util::find(m_inEdges, edge);
    ADE_ASSERT(m_inEdges.end() != it);
    util::unstable_erase(m_inEdges, it);
}

void Node::addOutEdge(Edge* edge)
{
    ADE_ASSERT(nullptr != edge);
    ADE_ASSERT(m_outEdges.end() == util::find(m_outEdges, edge));
    m_outEdges.emplace_back(edge);
}

void Node::removeOutEdge(Edge* edge)
{
    ADE_ASSERT(nullptr != edge);
    // Nodes usually have only a small amount of connections so linear search should be fine here
    auto it = util::find(m_outEdges, edge);
    ADE_ASSERT(m_outEdges.end() != it);
    util::unstable_erase(m_outEdges, it);
}

Graph* Node::getParent() const
{
    return m_parent;
}

Node::EdgeSetRange Node::inEdges()
{
    return util::map<HandleMapper>(util::toRange(m_inEdges));
}

Node::EdgeSetCRange Node::inEdges() const
{
    return util::map<HandleMapper>(util::toRange(m_inEdges));
}

Node::EdgeSetRange Node::outEdges()
{
    return util::map<HandleMapper>(util::toRange(m_outEdges));
}

Node::EdgeSetCRange Node::outEdges() const
{
    return util::map<HandleMapper>(util::toRange(m_outEdges));
}

Node::InNodeSetRange Node::inNodes()
{
    return util::map<InEdgeMapper>(inEdges());
}

Node::InNodeSetCRange Node::inNodes() const
{
    return util::map<InEdgeMapper>(inEdges());
}

Node::OutNodeSetRange Node::outNodes()
{
    return util::map<OutEdgeMapper>(outEdges());
}

Node::OutNodeSetCRange Node::outNodes() const
{
    return util::map<OutEdgeMapper>(outEdges());
}

EdgeHandle Node::HandleMapper::operator()(Edge* obj) const
{
    ADE_ASSERT(nullptr != obj);
    return Graph::HandleMapper()(obj);
}

NodeHandle Node::InEdgeMapper::operator()(const EdgeHandle& handle) const
{
    ADE_ASSERT(nullptr != handle);
    return handle->srcNode();
}

NodeHandle Node::OutEdgeMapper::operator()(const EdgeHandle& handle) const
{
    ADE_ASSERT(nullptr != handle);
    return handle->dstNode();
}

}
