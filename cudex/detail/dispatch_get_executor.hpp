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

#include "type_traits/is_detected.hpp"
#include "prologue.hpp"

#include <utility>


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using get_executor_member_function_t = decltype(std::declval<T>().get_executor());

template<class T>
using has_get_executor_member_function = is_detected<get_executor_member_function_t, T>;


template<class T>
using get_executor_free_function_t = decltype(get_executor(std::declval<T>()));

template<class T>
using has_get_executor_free_function = is_detected<get_executor_free_function_t, T>;


// dispatch case 1: arg.get_executor() exists
template<class T,
         CUDEX_REQUIRES(has_get_executor_member_function<T&&>::value)
        >
CUDEX_ANNOTATION
auto dispatch_get_executor(T&& arg)
  -> decltype(std::forward<T>(arg).get_executor())
{
  return std::forward<T>(arg).get_executor();
}


// dispatch case 2: get_executor(arg) exists
template<class T,
         CUDEX_REQUIRES(!has_get_executor_member_function<T&&>::value),
         CUDEX_REQUIRES(has_get_executor_free_function<T&&>::value)
        >
CUDEX_ANNOTATION
auto dispatch_get_executor(T&& arg)
  -> decltype(get_executor(std::forward<T>(arg)))
{
  return get_executor(std::forward<T>(arg));
}


template<class T>
using dispatch_get_executor_t = decltype(detail::dispatch_get_executor(std::declval<T>()));

template<class T>
using can_dispatch_get_executor = is_detected<dispatch_get_executor_t, T>;


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "epilogue.hpp"

