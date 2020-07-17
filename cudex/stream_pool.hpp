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

#include "detail/prologue.hpp"

#include <atomic>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <utility>
#include <vector>
#include "detail/event.hpp"
#include "detail/terminate.hpp"
#include "detail/throw_on_error.hpp"
#include "detail/type_traits/is_invocable.hpp"
#include "detail/with_current_device.hpp"
#include "executor/kernel_executor.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


class static_stream_pool
{
  public:
    // static_stream_pool::executor_type is just like kernel_executor
    // except that execute cannot be called from __device__ code
    class executor_type : private kernel_executor
    {
      public:
        executor_type(const executor_type&) = default;

        CUDEX_ANNOTATION
        inline bool operator==(const executor_type& other) const
        {
          return kernel_executor::operator==(other);
        }

        CUDEX_ANNOTATION
        inline bool operator!=(const executor_type& other) const
        {
          return !(*this == other);
        }

        using kernel_executor::device;
        using kernel_executor::coordinate_type;
        using kernel_executor::stream;

        // executor_type::execute can only be called on the host
        // because its stream was created on the host
        template<class Function,
                 CUDEX_REQUIRES(detail::is_invocable<Function>::value),
                 CUDEX_REQUIRES(std::is_trivially_copyable<Function>::value)
                >
        void execute(Function f) const
        {
          kernel_executor::execute(f);
        }

        // executor_type::bulk_execute can only be called on the host
        // because its stream was created on the host
        template<class Function,
                 CUDEX_REQUIRES(detail::is_invocable<Function,coordinate_type>::value),
                 CUDEX_REQUIRES(std::is_trivially_copyable<Function>::value)
                >
        void bulk_execute(Function f, coordinate_type shape) const
        {
          kernel_executor::bulk_execute(f, shape);
        }

      private:
        executor_type(cudaStream_t s, int d) : kernel_executor{s,d} {}
        friend class static_stream_pool;
    };

    inline static_stream_pool(int device, std::size_t num_streams)
      : streams_(make_streams(device, num_streams)),
        counter_{}
    {}

    static_stream_pool(const static_stream_pool&) = delete;

    inline ~static_stream_pool() noexcept
    {
      wait();
    }

    inline void wait()
    {
      std::vector<detail::event> events(streams_.begin(), streams_.end());
      
      for(const auto& event : events)
      {
        detail::throw_on_error(cudaEventSynchronize(event.native_handle()), "static_stream_pool::wait: CUDA error after cudaEventSynchronize");
      }
    }

    inline executor_type executor()
    {
      auto sv = stream();
      return {sv.native_handle(), sv.device()};
    }

  private:
    inline static std::vector<detail::stream> make_streams(int device, std::size_t num_streams)
    {
      std::vector<detail::stream> result;

      for(std::size_t i = 0; i < num_streams; ++i)
      {
        result.emplace_back(device);
      }

      return result;
    }

    inline detail::stream_view stream()
    {
      // round-robin through streams
      std::size_t i = (counter_++) % streams_.size();
      return streams_[i].view();
    }

    // XXX these need to be in managed memory
    // XXX this needn't be a vector because we don't need resize
    const std::vector<detail::stream> streams_;
    std::atomic<std::size_t> counter_;
};


CUDEX_NAMESPACE_CLOSE_BRACE

#include "detail/epilogue.hpp"

