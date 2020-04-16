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


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is an RAII type for cudaEvent_t
class event
{
  public:
    // this ctor isn't explicit to make it easy to construct a vector of these from a range of streams
    CUDEX_ANNOTATION
    inline event(cudaStream_t s)
      : native_handle_(make_event())
    {
      record_on(s);
    }

    CUDEX_ANNOTATION
    inline event(const stream& s)
      : event(s.native_handle())
    {}

    CUDEX_ANNOTATION
    inline event()
      : native_handle_{}
    {}

    CUDEX_ANNOTATION
    event(event&& other)
      : native_handle_{}
    {
      native_handle_ = other.native_handle_;
      other.native_handle_ = {};
    }

    event(const event&) = delete;

    CUDEX_ANNOTATION
    inline ~event() noexcept
    {
      if(native_handle_)
      {
#if CUDEX_HAS_CUDART
        detail::throw_on_error(cudaEventDestroy(native_handle()), "detail::event::~event: CUDA error after cudaEventDestroy");
#else
        detail::terminate_with_message("detail::event::~event: cudaEventDestroy is unavailable.");
#endif
      }
    }

    CUDEX_ANNOTATION
    event& operator=(event&& other)
    {
      cudaEvent_t tmp = native_handle_;
      native_handle_ = other.native_handle_;
      other.native_handle_ = tmp;
      return *this;
    }

    CUDEX_ANNOTATION
    cudaEvent_t native_handle() const
    {
      return native_handle_;
    }

    CUDEX_ANNOTATION
    void record_on(cudaStream_t s)
    {
      if(!native_handle())
      {
        native_handle_ = make_event();
      }

#if (__CUDA_ARCH__ == 0) or CUDEX_HAS_CUDART
      detail::throw_on_error(cudaEventRecord(native_handle(), s), "detail::event::record_on: CUDA error after cudaEventRecord");
#else
      detail::throw_runtime_error("detail::event::record_on: cudaEventRecord is unavailable.");
#endif
    }

    void wait() const
    {
      if(native_handle())
      {
        detail::throw_on_error(cudaEventSynchronize(native_handle()), "detail::event::wait: CUDA error after cudaEventSynchronize");
      }
    }

  private:
    CUDEX_ANNOTATION
    inline static cudaEvent_t make_event()
    {
      cudaEvent_t result{};
#if (__CUDA_ARCH__ == 0) or CUDEX_HAS_CUDART
      detail::throw_on_error(cudaEventCreateWithFlags(&result, cudaEventDisableTiming), "detail::event::make_event: CUDA error after cudaEventCreateWithFlags");
#else
      detail::throw_runtime_error("detail::event::make_event: cudaEventCreateWithFlags is unavailable.");
#endif
      return result;
    }

    cudaEvent_t native_handle_;
};


} // end detail

CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

