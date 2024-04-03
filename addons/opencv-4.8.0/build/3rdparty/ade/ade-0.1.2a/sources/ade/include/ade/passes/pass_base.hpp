// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file pass_base.hpp

#ifndef ADE_PASS_BASE_HPP
#define ADE_PASS_BASE_HPP

namespace ade
{

class Graph;

template<typename...>
class TypedGraph;

namespace passes
{
struct PassContext
{
    Graph& graph;
};

template<typename... Types>
struct TypedPassContext
{
    TypedGraph<Types...> graph;

    TypedPassContext(PassContext& context):
        graph(context.graph) {}
};

}

}


#endif // ADE_PASS_BASE_HPP
