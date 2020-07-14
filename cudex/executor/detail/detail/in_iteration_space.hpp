// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../../../detail/prologue.hpp"

#include <tuple>
#include <type_traits>
#include "../../../detail/tuple.hpp"
#include "../../../detail/utility/index_sequence.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Index, class Shape,
         CUDEX_REQUIRES(std::is_integral<Index>::value)
        >
CUDEX_ANNOTATION
constexpr bool in_iteration_space(const Index& index, const Shape& shape)
{
  return index < shape;
}


template<class Index, class Shape,
         CUDEX_REQUIRES(!std::is_integral<Index>::value)
        >
CUDEX_ANNOTATION
constexpr bool in_iteration_space(const Index& index, const Shape& shape);


template<class Index, class Shape>
CUDEX_ANNOTATION
constexpr bool in_iteration_space_impl(const Index&, const Shape&, index_sequence<>)
{
  return true;
}


template<class Index, class Shape, std::size_t i0, std::size_t... is>
CUDEX_ANNOTATION
constexpr bool in_iteration_space_impl(const Index& index, const Shape& shape, index_sequence<i0,is...>)
{
  return detail::in_iteration_space(detail::get<i0>(index), detail::get<i0>(shape)) and detail::in_iteration_space_impl(index, shape, index_sequence<is...>{});
}


template<class Index, class Shape,
         CUDEX_REQUIRES_DEF(!std::is_integral<Index>::value)
        >
CUDEX_ANNOTATION
constexpr bool in_iteration_space(const Index& index, const Shape& shape)
{
  constexpr std::size_t num_axes = std::tuple_size<Shape>::value;

  return detail::in_iteration_space_impl(index, shape, make_index_sequence<num_axes>{});
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

