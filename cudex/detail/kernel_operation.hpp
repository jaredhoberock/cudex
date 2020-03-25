#pragma once

#include "prologue.hpp"

#include <type_traits>
#include "launch_kernel.hpp"
#include "throw_runtime_error.hpp"

CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function>
class kernel_operation
{
  public:
    CUDEX_ANNOTATION
    kernel_operation(Function f, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device) noexcept
      : f_(f), grid_dim_(grid_dim), block_dim_(block_dim), shared_memory_size_(shared_memory_size), stream_(stream), device_(device)
    {}

    kernel_operation(const kernel_operation&) = default;

    CUDEX_ANNOTATION
    void start()
    {
      detail::launch_kernel(f_, grid_dim_, block_dim_, shared_memory_size_, stream_, device_);
    }

    CUDEX_ANNOTATION
    cudaStream_t stream() const
    {
      return stream_;
    }

  private:
    Function f_;
    dim3 grid_dim_;
    dim3 block_dim_;
    std::size_t shared_memory_size_;
    cudaStream_t stream_;
    int device_;
};


template<class Function,
         CUDEX_REQUIRES(detail::is_invocable<Function>::value),
         CUDEX_REQUIRES(std::is_trivially_copyable<Function>::value)
        >
CUDEX_ANNOTATION
kernel_operation<Function> make_kernel_operation(Function f, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device) noexcept
{
  return {f, grid_dim, block_dim, shared_memory_size, stream, device};
}


template<class Function>
class recorded_kernel_operation
{
  public:
    CUDEX_ANNOTATION
    recorded_kernel_operation(Function f, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device, cudaEvent_t recording_event) noexcept
      : kernel_op_(f, grid_dim, block_dim, shared_memory_size, stream, device),
        event_{recording_event}
    {}

    recorded_kernel_operation(const recorded_kernel_operation&) = delete;

    CUDEX_ANNOTATION
    recorded_kernel_operation(recorded_kernel_operation&& other) noexcept
      : kernel_op_(std::move(other.kernel_op_)),
        event_(other.event_)
    {
      other.event_ = {};
    }

    CUDEX_ANNOTATION
    void start() &&
    {
      if(event_)
      {
         // start the kernel
         kernel_op_.start();

         // record the event
         detail::throw_on_error(cudaEventRecord(event_, kernel_op_.stream()), "detail::recorded_kernel_operation::start: CUDA error after cudaEventRecord");

         // invalidate our state
         event_ = {};
      }
      else
      {
        detail::throw_runtime_error("detail::recorded_kernel_operation::start: Invalid state.");
      }
    }

  private:
    kernel_operation<Function> kernel_op_;
    cudaEvent_t event_;
};


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

