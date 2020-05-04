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
#include "detail/default_bulk_execute.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F, class S>
using bulk_execute_member_function_t = decltype(std::declval<E>().bulk_execute(std::declval<F>(), std::declval<S>()));

template<class E, class F, class S>
using has_bulk_execute_member_function = is_detected<bulk_execute_member_function_t, E, F, S>;


template<class E, class F, class S>
using bulk_execute_free_function_t = decltype(bulk_execute(std::declval<E>(), std::declval<F>(), std::declval<S>()));

template<class E, class F, class S>
using has_bulk_execute_free_function = is_detected<bulk_execute_free_function_t, E, F, S>;


// this is the type of bulk_execute
struct dispatch_bulk_execute
{
  // try a member function
  CUDEX_EXEC_CHECK_DISABLE
  template<class E, class F, class S,
           CUDEX_REQUIRES(has_bulk_execute_member_function<E&&,F&&,S&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(E&& e, F&& f, S&& s) const ->
    decltype(std::forward<E>(e).bulk_execute(std::forward<F>(f), std::forward<S>(s)))
  {
    return std::forward<E>(e).bulk_execute(std::forward<F>(f), std::forward<S>(s));
  }

  // try a free function
  CUDEX_EXEC_CHECK_DISABLE
  template<class E, class F, class S,
           CUDEX_REQUIRES(!has_bulk_execute_member_function<E&&,F&&,S&&>::value and
                          has_bulk_execute_free_function<E&&,F&&,S&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(E&& e, F&& f, S&& s) const ->
    decltype(bulk_execute(std::forward<E>(e), std::forward<F>(f), std::forward<S>(s)))
  {
    return bulk_execute(std::forward<E>(e), std::forward<F>(f), std::forward<S>(s));
  }

  // default path
  CUDEX_EXEC_CHECK_DISABLE
  template<class E, class F, class S,
           CUDEX_REQUIRES(!has_bulk_execute_member_function<E&&,F&&,S&&>::value and
                          !has_bulk_execute_free_function<E&&,F&&,S&&>::value and
                          can_default_bulk_execute<E&&,F&&,S&&>::value)
          >
  CUDEX_ANNOTATION
  void operator()(E&& e, F&& f, S&& s) const
  {
    detail::default_bulk_execute(std::forward<E>(e), std::forward<F>(f), std::forward<S>(s));
  }
};


} // end detail


namespace
{


// define the bulk_execute customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& bulk_execute = detail::static_const<detail::dispatch_bulk_execute>::value;
#else
const __device__ detail::dispatch_bulk_execute bulk_execute;
#endif


} // end anonymous namespace


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

