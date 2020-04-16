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
#include "throw_on_error.hpp"
#include "throw_runtime_error.hpp"
#include "terminate.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function>
CUDEX_ANNOTATION
void with_current_device(int device, Function&& f)
{
  int old_device = -1;

#if CUDEX_HAS_CUDART
  detail::throw_on_error(cudaGetDevice(&old_device), "detail::with_current_device: CUDA error after cudaGetDevice");
#else
  detail::throw_runtime_error("detail::with_current_device: cudaGetDevice is unavailable.");
#endif

  if(device != old_device)
  {
#ifdef __CUDA_ARCH__
    detail::terminate_with_message("detail::with_current_device: Requested device cannot differ from current device in __device__ code.");
#else
    detail::throw_on_error(cudaSetDevice(device), "detail::with_current_device: CUDA error after cudaSetDevice");
#endif
  }

  std::forward<Function>(f)();

  if(device != old_device)
  {
#ifndef __CUDA_ARCH__
    detail::throw_on_error(cudaSetDevice(old_device), "detail::with_current_device: CUDA error after cudaSetDevice");
#endif
  }
};


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

