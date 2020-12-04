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
#include "stream.hpp"
#include "throw_on_error.hpp"
#include "throw_runtime_error.hpp"
#include "with_current_device.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is an RAII type for cudaEvent_t
// it always refers to a valid cudaEvent_t resource.
// i.e., native_handle() is never equal to cudaEvent_t{}
class event
{
  public:
    CUDEX_ANNOTATION
    inline event(int device)
      : device_{device},
        native_handle_{make_event(device)},
        origin_{current_execution_space()}
    {}

    CUDEX_ANNOTATION
    inline event(int device, cudaStream_t s)
      : event{device}
    {
      record_on(s);
    }

    CUDEX_ANNOTATION
    inline event(const stream& s)
      : event(s.device(), s.native_handle())
    {}

    CUDEX_ANNOTATION
    event(event&& other)
      : event{0}
    {
      swap(other);
    }

    event(const event&) = delete;

    CUDEX_ANNOTATION
    inline ~event() noexcept
    {
      if(origin_ == current_execution_space())
      {
#if CUDEX_HAS_CUDART
        detail::throw_on_error(cudaEventDestroy(native_handle()), "detail::event::~event: CUDA error after cudaEventDestroy");
#else
        detail::terminate_with_message("detail::event::~event: cudaEventDestroy is unavailable.");
#endif
      }
      else
      {
#ifndef __CUDA_ARCH__
        std::cerr << "Warning: detail::event::~event: Leaking cudaEvent_t created in different execution space." << std::endl;
#else
        printf("Warning: detail::event::~event: Leaking cudaEvent_t created in different execution space.\n");
#endif
      }
    }

    CUDEX_ANNOTATION
    event& operator=(event&& other)
    {
      swap(other);
      return *this;
    }

    CUDEX_ANNOTATION
    int device() const
    {
      return device_;
    }

    CUDEX_ANNOTATION
    cudaEvent_t native_handle() const
    {
      return native_handle_;
    }

    CUDEX_ANNOTATION
    void record_on(cudaStream_t s)
    {
#if (__CUDA_ARCH__ == 0) or CUDEX_HAS_CUDART
      detail::throw_on_error(cudaEventRecord(native_handle(), s), "detail::event::record_on: CUDA error after cudaEventRecord");
#else
      detail::throw_runtime_error("detail::event::record_on: cudaEventRecord is unavailable.");
#endif
    }

    CUDEX_ANNOTATION
    bool is_ready() const
    {
      bool result = false;

#if (__CUDA_ARCH__ == 0) or CUDEX_HAS_CUDART
      cudaError_t status = cudaEventQuery(native_handle());

      if(status != cudaErrorNotReady and status != cudaSuccess)
      {
        detail::throw_on_error(status, "detail::event::is_ready: CUDA error after cudaEventQuery");
      }

      result = (status == cudaSuccess);
#else
      detail::throw_runtime_error("detail::event::record_on: cudaEventRecord is unavailable.");
#endif

      return result;
    }

    CUDEX_ANNOTATION
    void wait() const
    {
#if CUDEX_HAS_CUDART
#  ifndef __CUDA_ARCH__
      detail::throw_on_error(cudaEventSynchronize(native_handle()), "detail::event::wait: CUDA error after cudaEventSynchronize");
#  else
      detail::throw_on_error(cudaDeviceSynchronize(), "detail::event::wait: CUDA error after cudaDeviceSynchronize");
#  endif
#else
      detail::throw_runtime_error("detail::event::wait: Unsupported operation.");
#endif
    }

    CUDEX_ANNOTATION
    void swap(event& other)
    {
      int tmp_device = device_;
      device_ = other.device_;
      other.device_ = tmp_device;

      cudaEvent_t tmp_native_handle = native_handle_;
      native_handle_ = other.native_handle_;
      other.native_handle_ = tmp_native_handle;

      from tmp_origin = origin_;
      origin_ = other.origin_;
      other.origin_ = tmp_origin;
    }

  private:
    enum class from {host, device};

    CUDEX_ANNOTATION
    inline static from current_execution_space() noexcept
    {
#ifdef __CUDA_ARCH__
      return from::device;
#else
      return from::host;
#endif
    }

    CUDEX_ANNOTATION
    inline static cudaEvent_t make_event(int device)
    {
      cudaEvent_t result{};

      detail::with_current_device(device, [&result]
      {
#if (__CUDA_ARCH__ == 0) or CUDEX_HAS_CUDART
        detail::throw_on_error(cudaEventCreateWithFlags(&result, cudaEventDisableTiming), "detail::event::make_event: CUDA error after cudaEventCreateWithFlags");
#else
        detail::throw_runtime_error("detail::event::make_event: cudaEventCreateWithFlags is unavailable.");
#endif
      });

      return result;
    }

    int device_;
    cudaEvent_t native_handle_;
    from origin_;
};


} // end detail

CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

