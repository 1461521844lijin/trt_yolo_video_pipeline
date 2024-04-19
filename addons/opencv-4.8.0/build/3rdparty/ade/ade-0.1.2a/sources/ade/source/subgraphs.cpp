// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ade/helpers/subgraphs.hpp"

#include "ade/util/hash.hpp"

namespace ade
{

namespace
{
struct Subgraph final
{
    /// Nodes in current subgraph
    std::vector<NodeHandle> nodes;

    /// Nodes in current subgraph (needed for fast search)
    subgraphs::NodesSet acceptedNodes;

    /// Nodes rejected from current subgraph
    subgraphs::NodesSet rejectedNodes;

    void acceptNode(const NodeHandle& node)
    {
        ADE_ASSERT(nullptr != node);
        ADE_ASSERT(!util::contains(acceptedNodes, node));
        ADE_ASSERT(!util::contains(rejectedNodes, node));
        nodes.push_back(node);
        acceptedNodes.insert(node);
    }

    void rejectNode(const NodeHandle& node)
    {
        ADE_ASSERT(nullptr != node);
        ADE_ASSERT(!util::contains(acceptedNodes, node));
        ADE_ASSERT(!util::contains(rejectedNodes, node));
        rejectedNodes.insert(node);
    }

    void rollback()
    {
        ADE_ASSERT(!nodes.empty());
        auto node = nodes.back();
        ADE_ASSERT(util::contains(acceptedNodes, node));
        ADE_ASSERT(!util::contains(rejectedNodes, node));
        rejectedNodes.insert(node);
        acceptedNodes.erase(node);
        nodes.pop_back();
    }

    bool nodeVisited(const NodeHandle& node) const
    {
        ADE_ASSERT(nullptr != node);
        return util::contains(acceptedNodes, node) ||
               util::contains(rejectedNodes, node);
    }

    bool nodeRejected(const NodeHandle& node) const
    {
        ADE_ASSERT(nullptr != node);
        return util::contains(rejectedNodes, node);
    }

    bool empty() const
    {
        return nodes.empty();
    }
};

template<typename Visitor>
void visitAdjacent(const NodeHandle& node, Visitor&& visitor)
{
    ADE_ASSERT(nullptr != node);
    for (const auto& prevEdge : node->inEdges())
    {
        ADE_ASSERT(nullptr != prevEdge);
        if (visitor(prevEdge, SubgraphMergeDirection::In))
        {
            auto prevNode = prevEdge->srcNode();
            ADE_ASSERT(nullptr != prevNode);
            visitAdjacent(prevNode, std::forward<Visitor>(visitor));
        }
    }
    for (const auto& nextEdge : node->outEdges())
    {
        ADE_ASSERT(nullptr != nextEdge);
        if (visitor(nextEdge, SubgraphMergeDirection::Out))
        {
            auto nextNode = nextEdge->dstNode();
            ADE_ASSERT(nullptr != nextNode);
            visitAdjacent(nextNode, std::forward<Visitor>(visitor));
        }
    }
}

template<typename Visitor>
void visitPaths(std::vector<NodeHandle>& path, const NodeHandle& node,
                Visitor&& visitor)
{
    ADE_ASSERT(nullptr != node);
    for (const auto& nextNode : node->outNodes())
    {
        ADE_ASSERT(nullptr != nextNode);
        path.push_back(nextNode);
        if (visitor(path, node, nextNode))
        {
            visitPaths(path, nextNode, std::forward<Visitor>(visitor));
        }
        path.pop_back();
    }
}

}

std::vector<NodeHandle> assembleSubgraph(
    const NodeHandle& root,
    util::func_ref<bool(const EdgeHandle&,
                        SubgraphMergeDirection)> mergeChecker,
    util::func_ref<bool(const subgraphs::NodesSet& acceptedNodes,
                        const subgraphs::NodesSet& rejectedNodes)> topoChecker)
{
    ADE_ASSERT(nullptr != root);

    Subgraph subgraph;
    subgraph.acceptNode(root);

    visitAdjacent(root, [&](const EdgeHandle& edge,
                            SubgraphMergeDirection direction)
    {
        ADE_ASSERT(nullptr != edge);
        auto srcNode = getSrcMergeNode(edge, direction);
        auto dstNode = getDstMergeNode(edge, direction);
        ADE_ASSERT(nullptr != srcNode);
        ADE_ASSERT(nullptr != dstNode);
        if (subgraph.nodeRejected(srcNode) ||
            subgraph.nodeVisited(dstNode))
        {
            return false;
        }

        bool merge = mergeChecker(edge, direction);
        if (merge)
        {
            subgraph.acceptNode(dstNode);
        }
        else
        {
            subgraph.rejectNode(dstNode);
        }

        while(!topoChecker(subgraph.acceptedNodes, subgraph.rejectedNodes))
        {
            merge = false;
            if (subgraph.empty())
            {
                return false;
            }
            subgraph.rollback();
        }
        return merge;
    });

    return std::move(subgraph.nodes);
}

void findPaths(const NodeHandle& src, const NodeHandle& dst,
               util::func_ref<bool(const std::vector<NodeHandle>&)> visitor)
{
    ADE_ASSERT(nullptr != src);
    ADE_ASSERT(nullptr != dst);
    std::vector<NodeHandle> pathVec{src};
    bool found = false;
    (void)found; // Silence klocwork warning
    visitPaths(pathVec, src, [&](
               const std::vector<NodeHandle>& path,
               const NodeHandle& prev, const NodeHandle& next)
    {
        ADE_UNUSED(prev);
        ADE_ASSERT(nullptr != prev);
        ADE_ASSERT(nullptr != next);
        if (next == dst)
        {
            found = visitor(path);
            return false;
        }
        return !found;
    });
}
namespace
{

bool hasIntersectionImpl(const subgraphs::NodesSet& nodes1,
                         const subgraphs::NodesSet& nodes2)
{
    for (auto&& node : nodes1)
    {
        ADE_ASSERT(nullptr != node);
        if (util::contains(nodes2, node))
        {
            return true;
        }
    }
    return false;
}
bool hasIntersection(const subgraphs::NodesSet& nodes1,
                     const subgraphs::NodesSet& nodes2)
{
    if (nodes1.size() < nodes2.size())
    {
        return hasIntersectionImpl(nodes1, nodes2);
    }
    else
    {
        return hasIntersectionImpl(nodes2, nodes1);
    }
}
}

bool SubgraphSelfReferenceChecker::operator()(
        const subgraphs::NodesSet& acceptedNodes,
        const subgraphs::NodesSet& rejectedNodes)
{
    for (auto&& srcNode : acceptedNodes)
    {
        ADE_ASSERT(nullptr != srcNode);
        if (srcNode->outEdges().size() > 1)
        {
            for (auto&& dstNode : acceptedNodes)
            {
                ADE_ASSERT(nullptr != dstNode);
                if (srcNode == dstNode)
                {
                    continue;
                }

                if (dstNode->inEdges().size() > 1)
                {
                    auto key = std::make_pair(srcNode, dstNode);
                    if (!util::contains(m_cache, key))
                    {
                        updateCache(key);
                        ADE_ASSERT(util::contains(m_cache, key));
                    }

                    auto& paths = m_cache.find(key)->second;

                    if (hasIntersection(paths, rejectedNodes))
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

void SubgraphSelfReferenceChecker::reset()
{
    m_cache.clear();
}

std::size_t SubgraphSelfReferenceChecker::Hasher::operator()(
        const std::pair<NodeHandle, NodeHandle>& value) const
{
    ADE_ASSERT(nullptr != value.first);
    ADE_ASSERT(nullptr != value.second);

    ade::HandleHasher<ade::Node> h;
    return util::hash_combine(h(value.first), h(value.second));
}

void SubgraphSelfReferenceChecker::updateCache(
        const std::pair<NodeHandle, NodeHandle>& key)
{
    ADE_ASSERT(nullptr != key.first);
    ADE_ASSERT(nullptr != key.second);
    subgraphs::NodesSet nodes;
    auto it = m_connections.find(key.first);
    if (m_connections.end() != it)
    {
        for (auto node : it->second)
        {
            ADE_ASSERT(nullptr != node);
            if (key.second != node)
            {
                auto it2 = m_connections.find(node);
                if (m_connections.end() != it2 &&
                    util::contains(it2->second, key.second))
                {
                    nodes.insert(node);
                }
            }
        }
    }
    m_cache.insert({key, std::move(nodes)});
}

}
