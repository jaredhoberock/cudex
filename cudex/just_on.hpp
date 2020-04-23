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

#include "detail/prologue.hpp"

#include <utility>
#include "chaining_sender.hpp"
#include "detail/combinators/just_on/dispatch_just_on.hpp"
#include "detail/static_const.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of just_on
struct just_on_customization_point
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class E, class T,
           CUDEX_REQUIRES(can_dispatch_just_on<const E&,T&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr ensure_chaining_sender_t<dispatch_just_on_t<const E&,T&&>>
    operator()(const E& ex, T&& value) const
  {
    return CUDEX_NAMESPACE::ensure_chaining_sender(detail::dispatch_just_on(ex, std::forward<T>(value)));
  }
};


} // end detail


namespace
{


// define the just_on customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& just_on = detail::static_const<detail::just_on_customization_point>::value;
#else
const __device__ detail::just_on_customization_point just_on;
#endif


} // end anonymous namespace


template<class E, class T>
using just_on_t = decltype(CUDEX_NAMESPACE::just_on(std::declval<E>(), std::declval<T>()));


CUDEX_NAMESPACE_CLOSE_BRACE


#include "detail/epilogue.hpp"

