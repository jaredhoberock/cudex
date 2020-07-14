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


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


// XXX this is pseudocode for next_index
//     something like constexpr for is required to actually implement it this way
//template<class Index, class Shape>
//CUDEX_ANNOTATION
//void next_index(Index& index, const Shape& shape)
//{
//  constexpr Index origin{};
//
//  for(std::size_t dimension = std::tuple_size<Index>::value; dimension-- > 0;)
//  {
//    // recurse into this dimension
//    detail::next_index(detail::get<dimension>(index), detail::get<dimension>(shape));
//
//    if(detail::get<dimension>(index) < detail::get<dimension>(shape))
//    {
//      return;
//    }
//    else if(dimension > 0)
//    {
//      // don't roll the final dimension over to the origin
//      detail::get<dimension>(index) = detail::get<dimension>(origin);
//    }
//  }
//}


// integral index case
template<class Index, class Shape,
         CUDEX_REQUIRES(std::is_integral<Index>::value)
        >
CUDEX_ANNOTATION
void next_index(Index& index, const Shape&)
{
  ++index;
}


// tuple-like index case
template<class Index, class Shape,
         CUDEX_REQUIRES(!std::is_integral<Index>::value)
        >
CUDEX_ANNOTATION
void next_index(Index& index, const Shape& shape);


template<class Index, class Shape>
CUDEX_ANNOTATION
void next_index_impl(std::integral_constant<std::size_t,0>, Index& index, const Shape& shape)
{
  // terminal case -- dimension zero

  // recurse into this dimension
  detail::next_index(detail::get<0>(index), detail::get<0>(shape));

  // don't roll dimension zero over to the origin
}


template<std::size_t dimension, class Index, class Shape>
CUDEX_ANNOTATION
void next_index_impl(std::integral_constant<std::size_t,dimension>, Index& index, const Shape& shape)
{
  constexpr Index origin{};

  // recurse into this dimension
  detail::next_index(detail::get<dimension>(index), detail::get<dimension>(shape));

  if(detail::get<dimension>(index) < detail::get<dimension>(shape))
  {
    // break the iteration
    return;
  }

  // don't roll this dimension over to the origin
  detail::get<dimension>(index) = detail::get<dimension>(origin);

  // continue iterating with dimension-1
  detail::next_index_impl(std::integral_constant<std::size_t,dimension-1>{}, index, shape);
}


template<class Index, class Shape,
         CUDEX_REQUIRES_DEF(!std::is_integral<Index>::value)
        >
CUDEX_ANNOTATION
void next_index(Index& index, const Shape& shape)
{
  constexpr std::size_t dimension = std::tuple_size<Index>::value - 1;
  detail::next_index_impl(std::integral_constant<std::size_t,dimension>{}, index, shape);
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../../../detail/epilogue.hpp"

