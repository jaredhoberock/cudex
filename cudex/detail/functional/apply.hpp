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

#include "../prologue.hpp"

#include <cstdint>
#include <tuple>
#include <type_traits>
#include "../utility/index_sequence.hpp"
#include "../tuple.hpp"
#include "../type_traits.hpp"
#include "invoke.hpp"

CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{
namespace apply_detail
{


template<class Invocable, class Tuple, std::size_t... Indices>
CUDEX_ANNOTATION
auto apply_impl(Invocable&& f, Tuple&& args, detail::index_sequence<Indices...>)
  -> decltype(detail::invoke(std::forward<Invocable>(f), detail::get<Indices>(std::forward<Tuple>(args))...))
{
  return detail::invoke(std::forward<Invocable>(f), detail::get<Indices>(std::forward<Tuple>(args))...);
}


template<class Tuple>
using tuple_indices_t = detail::make_index_sequence<std::tuple_size<decay_t<Tuple>>::value>;


} // end apply_detail


template<class Invocable, class Tuple>
using apply_result_t = decltype(apply_detail::apply_impl(std::declval<Invocable>(), std::declval<Tuple>(), apply_detail::tuple_indices_t<Tuple>{}));


template<class Invocable, class Tuple>
using is_applicable = is_detected<apply_result_t, Invocable, Tuple>;


template<class Invocable, class Tuple,
         CUDEX_REQUIRES(is_applicable<Invocable&&,Tuple&&>::value)
        >
CUDEX_ANNOTATION
apply_result_t<Invocable&&,Tuple&&> apply(Invocable&& f, Tuple&& args)
{
  return apply_detail::apply_impl(std::forward<Invocable>(f), std::forward<Tuple>(args), apply_detail::tuple_indices_t<Tuple>{});
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

