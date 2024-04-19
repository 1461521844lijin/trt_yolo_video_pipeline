// Copyright (C) 2018 Intel Corporation
//
//
// SPDX-License-Identifier: Apache-2.0
//

/// @file map_range.hpp

#ifndef ADE_UTIL_MAP_RANGE_HPP
#define ADE_UTIL_MAP_RANGE_HPP

#include <type_traits>
#include <utility>
#include <tuple>

#include "ade/util/type_traits.hpp"
#include "ade/util/range.hpp"
#include "ade/util/assert.hpp"
#include "ade/util/range_iterator.hpp"

namespace ade
{
namespace util
{

inline namespace Range
{

template<typename PrevRange, typename Mapper>
struct MapRange : public IterableRange<MapRange<PrevRange, Mapper>>
{
    PrevRange prevRange;
    Mapper mapper; // TODO: implement size optimization for empty objects

    MapRange() = default;
    MapRange(const MapRange&) = default;
    MapRange(MapRange&&) = default;
    MapRange& operator=(const MapRange&) = default;
    MapRange& operator=(MapRange&&) = default;
    template<typename PR, typename M>
    MapRange(PR&& pr, M&& m):
        prevRange(std::forward<PR>(pr)),
        mapper(std::forward<M>(m)) {}

    bool empty() const
    {
        return prevRange.empty();
    }

    void popFront()
    {
        ADE_ASSERT(!empty());
        prevRange.popFront();
    }

    auto front() -> typename std::decay<decltype(mapper(prevRange.front()))>::type
    {
        ADE_ASSERT(!empty());
        return mapper(prevRange.front());
    }

    auto front() const -> typename std::decay<decltype(mapper(prevRange.front()))>::type
    {
        ADE_ASSERT(!empty());
        return mapper(prevRange.front());
    }

    template<typename R = PrevRange,
             util::enable_b_t<details::has_size_fun<R>::value> = true> //SFINAE
    auto size() const
    ->decltype(details::range_size_wrapper(std::declval<R>()))
    {
        return details::range_size_wrapper(prevRange);
    }
};

template<typename PrevRange, typename Mapper>
inline auto size(const MapRange<PrevRange, Mapper>& range)
->decltype(range.size())
{
    return range.size();
}

template<typename Mapper, typename PrevRange>
MapRange<PrevRange, Mapper> map(PrevRange&& prevRange, Mapper&& mapper)
{
    return MapRange<PrevRange, Mapper>(std::forward<PrevRange>(prevRange), std::forward<Mapper>(mapper));
}

template<typename Mapper, typename PrevRange>
MapRange<PrevRange, Mapper> map(PrevRange&& prevRange)
{
    return MapRange<PrevRange, Mapper>(std::forward<PrevRange>(prevRange), Mapper());
}

}
}
} // namespace ade

#endif // ADE_UTIL_MAP_RANGE_HPP
