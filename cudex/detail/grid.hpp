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

#include "prologue.hpp"

#include <cuda_runtime_api.h>
#include <utility>
#include "kernel_launch.hpp"
#include "throw_on_error.hpp"
#include "throw_runtime_error.hpp"

CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function>
class grid
{
  private:
    static_assert(std::is_trivially_copyable<Function>::value, "Function must be trivially copyable.");

    class launch_kernel_and_record_event
    {
      public:
        CUDEX_ANNOTATION
        launch_kernel_and_record_event(kernel_launch<Function> kernel, cudaEvent_t successor)
          : kernel_launch_(std::move(kernel)), successor_(successor)
        {}

        CUDEX_ANNOTATION
        void start() &&
        {
          if(valid())
          {
            // start the kernel
            kernel_launch_.start();

            // record the event
            detail::throw_on_error(cudaEventRecord(successor_, kernel_launch_.stream()), "grid::launch_kernel_and_record_event::start: CUDA error after cudaEventRecord");

            // invalidate our state
            successor_ = 0;
          }
          else
          {
            detail::throw_runtime_error("grid::launch_kernel_and_record_event::start: Invalid state.");
          }
        }

      private:
        CUDEX_ANNOTATION
        bool valid() const
        {
          return successor_ != 0;
        }

        kernel_launch<Function> kernel_launch_;
        cudaEvent_t successor_;
    };

  public:
    CUDEX_ANNOTATION
    grid(Function f, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device) noexcept
      : kernel_launch_{f, grid_dim, block_dim, shared_memory_size, stream, device}
    {}

    CUDEX_ANNOTATION
    launch_kernel_and_record_event connect(cudaEvent_t event) && noexcept
    {
      return {std::move(kernel_launch_), event};
    }

    // what should we actually do in general?
    // run the receiver in a host callback?
    // enqueue the receiver on a thread pool?
    //template<class Receiver,
    //         CUDEX_REQUIRES(is_receiver<Receiver>::value)
    //        >
    //void connect(Receiver&& r);

  private:
    kernel_launch<Function> kernel_launch_;
};


template<class Function>
CUDEX_ANNOTATION
grid<Function> make_grid(Function f, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device) noexcept
{
  return {f, grid_dim, block_dim, shared_memory_size, stream, device};
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

