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

#include <utility>
#include "../../type_traits/is_detected.hpp"
#include "default_just_on.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class T>
using just_on_member_function_t = decltype(std::declval<E>().just_on(std::declval<E>(), std::declval<T>()));

template<class E, class T>
using has_just_on_member_function = is_detected<just_on_member_function_t, E, T>;


template<class E, class T>
using just_on_free_function_t = decltype(just_on(std::declval<E>(), std::declval<T>()));

template<class E, class T>
using has_just_on_free_function = is_detected<just_on_free_function_t, E, T>;


// dispatch case 1: ex.just_on(value) exists
template<class Executor, class T,
         CUDEX_REQUIRES(has_just_on_member_function<const Executor&,T&&>::value)
        >
CUDEX_ANNOTATION
auto dispatch_just_on(const Executor& ex, T&& value)
  -> decltype(ex.just_on(std::forward<T>(value)))
{
  return ex.just_on(std::forward<T>(value));
}


// dispatch case 1: ex.just_on(f) does not exist
//                  just_on(ex, f) does exist
template<class Executor, class T,
         CUDEX_REQUIRES(!has_just_on_member_function<const Executor&,T&&>::value),
         CUDEX_REQUIRES(has_just_on_free_function<const Executor&,T&&>::value)
        >
CUDEX_ANNOTATION
auto dispatch_just_on(const Executor& ex, T&& value)
  -> decltype(just_on(ex, std::forward<T>(value)))
{
  return just_on(ex, std::forward<T>(value));
}


// dispatch case 2: ex.just_on(f) does not exist
//                  just_on(ex, f) does not exist
template<class Executor, class T,
         CUDEX_REQUIRES(!has_just_on_member_function<const Executor&,T&&>::value),
         CUDEX_REQUIRES(!has_just_on_free_function<const Executor&,T&&>::value)
        >
CUDEX_ANNOTATION
constexpr auto dispatch_just_on(const Executor& ex, T&& value)
  -> decltype(detail::default_just_on(ex, std::forward<T>(value)))
{
  return detail::default_just_on(ex, std::forward<T>(value));
}


template<class E, class T>
using dispatch_just_on_t = decltype(detail::dispatch_just_on(std::declval<E>(), std::declval<T>()));

template<class E, class T>
using can_dispatch_just_on = is_detected<dispatch_just_on_t, E, T>;


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

