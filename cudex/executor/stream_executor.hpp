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

#include "../detail/prologue.hpp"

#include "../detail/launch_kernel.hpp"
#include "../detail/stream.hpp"
#include "is_executor.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


class stream_executor
{
  public:
    CUDEX_ANNOTATION
    inline stream_executor(cudaStream_t stream, int device)
      : stream_{device, stream}
    {}

    CUDEX_ANNOTATION
    inline explicit stream_executor(cudaStream_t stream)
      : stream_executor(stream, 0)
    {}

    CUDEX_ANNOTATION
    inline stream_executor()
      : stream_executor(cudaStream_t{0})
    {}

    stream_executor(const stream_executor&) = default;

    template<class Function,
             CUDEX_REQUIRES(std::is_trivially_copyable<Function>::value)
            >
    CUDEX_ANNOTATION
    void execute(Function f) const noexcept
    {
      detail::launch_kernel(f, dim3(1), dim3(1), 0, stream_.native_handle(), stream_.device());
    }

    CUDEX_ANNOTATION
    bool operator==(const stream_executor& other) const
    {
      return stream_ == other.stream_;
    }

    CUDEX_ANNOTATION
    bool operator!=(const stream_executor& other) const
    {
      return !(*this == other);
    }

    CUDEX_ANNOTATION
    cudaStream_t stream() const
    {
      return stream_.native_handle();
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

  private:
    detail::stream_view stream_;
};


static_assert(is_executor<stream_executor>::value, "Error.");


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

