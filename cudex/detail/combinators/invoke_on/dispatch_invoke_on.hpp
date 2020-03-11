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
#include "default_invoke_on.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F, class... Args>
using invoke_on_member_function_t = decltype(std::declval<E>().invoke_on(std::declval<F>(), std::declval<Args>()...));

template<class E, class F, class... Args>
using has_invoke_on_member_function = is_detected<invoke_on_member_function_t, E, F, Args...>;


template<class E, class F, class... Args>
using invoke_on_free_function_t = decltype(invoke_on(std::declval<E>(), std::declval<F>(), std::declval<Args>()...));

template<class E, class F, class... Args>
using has_invoke_on_free_function = is_detected<invoke_on_free_function_t, E, F, Args...>;


// dispatch case 1: ex.invoke_on(f) exists
template<class Executor, class Function, class... Args,
         CUDEX_REQUIRES(has_invoke_on_member_function<const Executor&,Function&&,Args&&...>::value)
        >
CUDEX_ANNOTATION
auto dispatch_invoke_on(const Executor& ex, Function&& f, Args&&... args)
  -> decltype(ex.invoke_on(std::forward<Function>(f), std::forward<Args>(args)...))
{
  return ex.invoke_on(std::forward<Function>(f), std::forward<Args>(args)...);
}


// dispatch case 1: ex.invoke_on(f) does not exist
//                  invoke_on(ex, f) does exist
template<class Executor, class Function, class... Args,
         CUDEX_REQUIRES(!has_invoke_on_member_function<const Executor&,Function&&,Args&&...>::value),
         CUDEX_REQUIRES(has_invoke_on_free_function<const Executor&,Function&&,Args&&...>::value)
        >
CUDEX_ANNOTATION
auto dispatch_invoke_on(const Executor& ex, Function&& f, Args&&... args)
  -> decltype(invoke_on(ex, std::forward<Function>(f), std::forward<Args>(args)...))
{
  return invoke_on(ex, std::forward<Function>(f), std::forward<Args>(args)...);
}


// dispatch case 2: ex.invoke_on(f) does not exist
//                  invoke_on(ex, f) does not exist
template<class Executor, class Function, class... Args,
         CUDEX_REQUIRES(!has_invoke_on_member_function<const Executor&,Function&&,Args&&...>::value),
         CUDEX_REQUIRES(!has_invoke_on_free_function<const Executor&,Function&&,Args&&...>::value)
        >
CUDEX_ANNOTATION
constexpr auto dispatch_invoke_on(const Executor& ex, Function&& f, Args&&... args)
  -> decltype(detail::default_invoke_on(ex, std::forward<Function>(f), std::forward<Args>(args)...))
{
  return detail::default_invoke_on(ex, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class E, class F, class... As>
using dispatch_invoke_on_t = decltype(detail::dispatch_invoke_on(std::declval<E>(), std::declval<F>(), std::declval<As>()...));

template<class E, class F, class... As>
using can_dispatch_invoke_on = is_detected<dispatch_invoke_on_t, E, F, As...>;


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "../../../detail/epilogue.hpp"

