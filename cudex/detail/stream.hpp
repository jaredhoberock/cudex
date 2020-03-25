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

#include "prologue.hpp"

#include <atomic>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "throw_on_error.hpp"
#include "terminate.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function>
CUDEX_ANNOTATION
void with_current_device(int device, Function&& f)
{
  int old_device{};

  detail::throw_on_error(cudaGetDevice(&old_device), "detail::with_current_device: CUDA error after cudaGetDevice");

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


// this is a non-owning view of a cudaStream_t
// it provides a convenient way to associate a device with a cudaStream_t
class stream_view
{
  public:
    CUDEX_ANNOTATION
    stream_view(int device, cudaStream_t native_handle)
      : device_(device),
        native_handle_(native_handle)
    {}

    stream_view(const stream_view&) = default;

    CUDEX_ANNOTATION
    cudaStream_t native_handle() const
    {
      return native_handle_;
    }

    CUDEX_ANNOTATION
    int device() const
    {
      return device_;
    }

    CUDEX_ANNOTATION
    bool operator==(const stream_view& other) const
    {
      return (device() == other.device()) and (native_handle() == other.native_handle());
    }

    CUDEX_ANNOTATION
    bool operator!=(const stream_view& other) const
    {
      return !(*this == other);
    }

    CUDEX_ANNOTATION
    void synchronize() const
    {
      detail::throw_on_error(cudaStreamSynchronize(native_handle_), "detail::stream_view::synchronize: CUDA error after cudaStreamSynchronize");
    }

  private:
    int device_;
    cudaStream_t native_handle_;
};


// this is an RAII type for cudaStream_t
class stream
{
  public:
    // this ctor isn't explicit to make it easy to construct a vector of these from a range of integers
    CUDEX_ANNOTATION
    stream(int device = 0)
      : stream_view_(make_stream(device))
    {}

    stream(const stream&) = delete;

    CUDEX_ANNOTATION
    inline ~stream() noexcept
    {
      detail::throw_on_error(cudaStreamDestroy(native_handle()), "static_stream_pool::stream::~stream: CUDA error after cudaStreamDestroy");
    }

    CUDEX_ANNOTATION
    cudaStream_t native_handle() const
    {
      return stream_view_.native_handle();
    }

    CUDEX_ANNOTATION
    int device() const
    {
      return stream_view_.device();
    }

  private:
    CUDEX_ANNOTATION
    inline static stream_view make_stream(int device)
    {
      cudaStream_t result{};

      with_current_device(device, [&result]
      {
        detail::throw_on_error(cudaStreamCreateWithFlags(&result, cudaStreamNonBlocking), "static_stream_pool::stream::make_stream: CUDA error after cudaStreamCreateWithFlags");
      });

      return {device, result};
    }

    stream_view stream_view_;
};


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

