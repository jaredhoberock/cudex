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

#include "../../prologue.hpp"

#include <utility>
#include "../../../invoke_on.hpp"
#include "../../type_traits/decay.hpp"
#include "../../execution.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
struct return_value
{
  T value;

  CUDEX_ANNOTATION
  T operator()()
  {
    return std::move(value);
  }
};


template<class T>
CUDEX_ANNOTATION
return_value<decay_t<T>> make_return_value(T&& value)
{
  return {std::forward<T>(value)};
}


template<class Executor, class T,
         CUDEX_REQUIRES(detail::execution::is_executor<Executor>::value)
        >
CUDEX_ANNOTATION
auto default_just_on(const Executor& ex, T&& value)
  -> decltype(CUDEX_NAMESPACE::invoke_on(ex, detail::make_return_value(std::forward<T>(value))))
{
  return CUDEX_NAMESPACE::invoke_on(ex, detail::make_return_value(std::forward<T>(value)));
}


} // end namespace detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "../../epilogue.hpp"

