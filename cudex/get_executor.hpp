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
#include "detail/dispatch_get_executor.hpp"
#include "detail/execution.hpp"
#include "detail/static_const.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of get_executor
struct get_executor_customization_point
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class T,
           CUDEX_REQUIRES(can_dispatch_get_executor<T&&>::value),
           CUDEX_REQUIRES(execution::is_executor<dispatch_get_executor_t<T&&>>::value)
          >
  CUDEX_ANNOTATION
  constexpr dispatch_get_executor_t<T&&> operator()(T&& arg) const
  {
    return detail::dispatch_get_executor(std::forward<T>(arg));
  }
};


} // end detail


namespace
{


// define the get_executor customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& get_executor = detail::static_const<detail::get_executor_customization_point>::value;
#else
const __device__ detail::get_executor_customization_point get_executor;
#endif


} // end anonymous namespace


CUDEX_NAMESPACE_CLOSE_BRACE


#include "detail/epilogue.hpp"

