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

#include "../detail/prologue.hpp"

#include <utility>
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"
#include "is_executor.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using executor_member_function_result_t = decltype(std::declval<T>().executor());

template<class T>
using has_executor_member_function = is_detected<executor_member_function_result_t, T>;


template<class T>
using get_executor_free_function_result_t = decltype(get_executor(std::declval<T>()));

template<class T>
using has_get_executor_free_function = is_detected<get_executor_free_function_result_t, T>;


// this is the type of get_executor
struct dispatch_get_executor
{
  // case 1: arg.executor() exists
  CUDEX_EXEC_CHECK_DISABLE
  template<class T,
           CUDEX_REQUIRES(has_executor_member_function<T&&>::value),
           CUDEX_REQUIRES(is_executor<executor_member_function_result_t<T&&>>::value)
          >
  CUDEX_ANNOTATION
  constexpr executor_member_function_result_t<T&&> operator()(T&& arg) const
  {
    return std::forward<T>(arg).executor();
  }


  // case 2: get_executor(arg) exists
  // we use "get_executor" when calling the free function in order to avoid
  // colliding with the concept named "executor"
  CUDEX_EXEC_CHECK_DISABLE
  template<class T,
           CUDEX_REQUIRES(!has_executor_member_function<T&&>::value),
           CUDEX_REQUIRES(has_get_executor_free_function<T&&>::value),
           CUDEX_REQUIRES(is_executor<get_executor_free_function_result_t<T&&>>::value)
          >
  CUDEX_ANNOTATION
  constexpr get_executor_free_function_result_t<T&&> operator()(T&& arg) const
  {
    return get_executor(std::forward<T>(arg));
  }
};


} // end detail


namespace
{


// define the get_executor customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& get_executor = detail::static_const<detail::dispatch_get_executor>::value;
#else
const __device__ detail::dispatch_get_executor get_executor;
#endif


} // end anonymous namespace


template<class T>
using get_executor_result_t = decltype(CUDEX_NAMESPACE::get_executor(std::declval<T>()));


CUDEX_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

