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

#include <cuda_runtime_api.h>
#include <type_traits>
#include "../detail/type_traits/is_detected.hpp" 
#include "../executor/is_executor.hpp" 
#include "detail/basic_executor_property.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using dynamic_shared_memory_size_member_function_t = decltype(std::declval<T>().dynamic_shared_memory_size());


} // end detail


class dynamic_shared_memory_size_property : 
  public detail::basic_executor_property<
    dynamic_shared_memory_size_property,
    true, // requireable
    true, // preferable
    std::size_t
  >
{
  public:
    CUDEX_ANNOTATION
    constexpr explicit dynamic_shared_memory_size_property(std::size_t value) : value_(value) {}

    CUDEX_ANNOTATION
    constexpr dynamic_shared_memory_size_property operator()(std::size_t value) const
    {
      return dynamic_shared_memory_size_property{value};
    }

    CUDEX_ANNOTATION
    constexpr std::size_t value() const
    {
      return value_;
    }

  private:
    CUDEX_EXEC_CHECK_DISABLE
    template<class Executor,
             CUDEX_REQUIRES(is_executor<Executor>::value),
             CUDEX_REQUIRES(detail::is_detected_exact<std::size_t, detail::dynamic_shared_memory_size_member_function_t, Executor>::value)
            >
    CUDEX_ANNOTATION
    friend cudaStream_t query(const Executor& ex, const dynamic_shared_memory_size_property&)
    {
      return ex.dynamic_shared_memory_size();
    }

    CUDEX_EXEC_CHECK_DISABLE
    template<class Executor,
             CUDEX_REQUIRES(is_executor<Executor>::value),
             CUDEX_REQUIRES(!detail::is_detected_exact<std::size_t, detail::dynamic_shared_memory_size_member_function_t, Executor>::value)
            >
    CUDEX_ANNOTATION
    friend std::size_t query(const Executor& ex, const dynamic_shared_memory_size_property&)
    {
      // by default, dynamic shared memory size is zero
      return 0;
    }

    std::size_t value_;
};


#ifndef __CUDA_ARCH__
static constexpr dynamic_shared_memory_size_property dynamic_shared_memory_size{0};
#else
const __device__ dynamic_shared_memory_size_property dynamic_shared_memory_size{0};
#endif


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

