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

#include <vector_types.h>
#include "../detail/functional/invoke.hpp"
#include "../detail/launch_kernel.hpp"
#include "../detail/stream.hpp"
#include "../detail/type_traits/is_invocable.hpp"
#include "../property/bulk_guarantee.hpp"
#include "../property/dynamic_shared_memory_size.hpp"
#include "is_device_executor.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function, class Coord>
struct invoke_with_builtin_indices
{
  mutable Function f;

  CUDEX_ANNOTATION
  void operator()() const
  {
#ifdef __CUDA_ARCH__
    detail::invoke(f, Coord{blockIdx, threadIdx});
#endif
  }
};


} // end detail


class kernel_executor
{
  public:
    CUDEX_ANNOTATION
    inline kernel_executor(cudaStream_t stream, std::size_t dynamic_shared_memory_size, int device)
      : stream_{device, stream},
        dynamic_shared_memory_size_{dynamic_shared_memory_size}
    {}


    CUDEX_ANNOTATION
    inline kernel_executor(cudaStream_t stream, int device)
      : kernel_executor{stream, 0, device}
    {}


    CUDEX_ANNOTATION
    inline explicit kernel_executor(cudaStream_t stream)
      : kernel_executor(stream, 0, 0)
    {}


    CUDEX_ANNOTATION
    inline kernel_executor()
      : kernel_executor(cudaStream_t{0})
    {}


    kernel_executor(const kernel_executor&) = default;


    template<class Function,
             CUDEX_REQUIRES(std::is_trivially_copyable<Function>::value),
             CUDEX_REQUIRES(detail::is_invocable<Function>::value)
            >
    CUDEX_ANNOTATION
    void execute(Function f) const
    {
      detail::launch_kernel(f, dim3(1), dim3(1), dynamic_shared_memory_size_, stream_.native_handle(), stream_.device());
    }


    // XXX TODO coordinate_type should have a tuple-like interface
    struct coordinate_type
    {
      ::dim3 block;
      ::dim3 thread;
    };


    template<class Function,
             CUDEX_REQUIRES(std::is_trivially_copyable<Function>::value),
             CUDEX_REQUIRES(detail::is_invocable<Function,coordinate_type>::value)
            >
    CUDEX_ANNOTATION
    void bulk_execute(Function f, coordinate_type shape) const
    {
      detail::launch_kernel(detail::invoke_with_builtin_indices<Function,coordinate_type>{f}, shape.block, shape.thread, dynamic_shared_memory_size_, stream_.native_handle(), stream_.device());
    }



    CUDEX_ANNOTATION
    bool operator==(const kernel_executor& other) const
    {
      return (stream_ == other.stream_) and (dynamic_shared_memory_size_ == other.dynamic_shared_memory_size_);
    }


    CUDEX_ANNOTATION
    bool operator!=(const kernel_executor& other) const
    {
      return !(*this == other);
    }


    CUDEX_ANNOTATION
    cudaStream_t stream() const
    {
      return stream_.native_handle();
    }


    CUDEX_ANNOTATION
    std::size_t dynamic_shared_memory_size() const
    {
      return dynamic_shared_memory_size_;
    }


    CUDEX_ANNOTATION
    kernel_executor require(dynamic_shared_memory_size_property request) const
    {
      return {stream(), request.value(), device()};
    }


    CUDEX_ANNOTATION
    void stream_wait_for(cudaEvent_t e) const
    {
      stream_.wait_for(e);
    }


    CUDEX_ANNOTATION
    int device() const
    {
      return stream_.device();
    }


    CUDEX_ANNOTATION
    constexpr static auto query(bulk_guarantee_t)
    {
      return bulk_guarantee.scoped(bulk_guarantee.parallel, bulk_guarantee.concurrent);
    }


  private:
    detail::stream_view stream_;
    std::size_t dynamic_shared_memory_size_;
};


static_assert(is_device_executor<kernel_executor>::value, "Error.");


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

