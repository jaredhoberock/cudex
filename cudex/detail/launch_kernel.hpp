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


#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <type_traits>
#include "throw_on_error.hpp"
#include "throw_runtime_error.hpp"
#include "type_traits/is_invocable.hpp"
#include "with_current_device.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function>
__global__ void global_function(Function f)
{
  f();
}


template<class Arg>
CUDEX_ANNOTATION
void workaround_unused_variable_warning(Arg&&) noexcept {}


template<class Function,
         CUDEX_REQUIRES(detail::is_invocable<Function>::value),
         CUDEX_REQUIRES(std::is_trivially_copyable<Function>::value)
        >
CUDEX_ANNOTATION
void launch_kernel(Function kernel, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device)
{
#if __CUDACC__
  detail::with_current_device(device, [=]() mutable
  {
    // point to the kernel
    void* ptr_to_kernel = reinterpret_cast<void*>(&detail::global_function<Function>);

    // reference the kernel to encourage the compiler not to optimize it away
    detail::workaround_unused_variable_warning(ptr_to_kernel);

#  if CUDEX_HAS_CUDART
    // ignore empty launches
    if(grid_dim.x * grid_dim.y * grid_dim.z * block_dim.x * block_dim.y * block_dim.z != 0)
    {
#    ifndef __CUDA_ARCH__
      // point to the parameter
      void* ptr_to_arg[] = {reinterpret_cast<void*>(&kernel)};

      // launch the kernel
      if(cudaError_t error = cudaLaunchKernel(ptr_to_kernel, grid_dim, block_dim, ptr_to_arg, shared_memory_size, stream))
      {
        detail::throw_on_error(error, "detail::launch_kernel: CUDA error after cudaLaunchKernel");
      }
#    else
      // copy the parameter
      void* ptr_to_arg = cudaGetParameterBuffer(std::alignment_of<Function>::value, sizeof(Function));
      std::memcpy(ptr_to_arg, &kernel, sizeof(Function));

      // launch the kernel
      if(cudaError_t error = cudaLaunchDevice(ptr_to_kernel, ptr_to_arg, grid_dim, block_dim, shared_memory_size, stream))
      {
        detail::throw_on_error(error, "detail::launch_kernel: CUDA error after cudaLaunchDevice");
      }
#    endif
      }
#  else
      detail::throw_runtime_error("detail::launch_kernel requires the CUDA Runtime.");
#  endif
  });
#else
  detail::throw_runtime_error("detail::launch_kernel requries CUDA C++ language support.");
#endif
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "epilogue.hpp"

