#pragma once

#include "prologue.hpp"

#include <cstring>
#include <cuda_runtime_api.h>
#include <type_traits>
#include "throw_on_error.hpp"

CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function>
__global__ void global_function(Function f)
{
  f();
}


template<class Function>
class kernel_launch
{
  public:
    static_assert(std::is_trivially_copyable<Function>::value, "Function must be trivially copyable.");

    CUDEX_ANNOTATION
    kernel_launch(Function f, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device) noexcept
      : f_(f), grid_dim_(grid_dim), block_dim_(block_dim), shared_memory_size_(shared_memory_size), stream_(stream), device_(device)
    {}

    kernel_launch(const kernel_launch&) = delete;

    kernel_launch(kernel_launch&&) = default;

    CUDEX_ANNOTATION
    void start()
    {
      // switch to the requested device
      int old_device = set_device(device_);

      // point to the kernel
      void* ptr_to_kernel = reinterpret_cast<void*>(&detail::global_function<Function>);

      // reference the kernel to encourage the compiler not to optimize it away
      workaround_unused_variable_warning(ptr_to_kernel);

#if CUDEX_HAS_CUDART
      // ignore empty launches
      if(grid_dim_.x * grid_dim_.y * grid_dim_.z * block_dim_.x * block_dim_.y * block_dim_.z != 0)
      {
#  ifndef __CUDA_ARCH__
        // point to the argument
        void* ptr_to_arg[] = {reinterpret_cast<void*>(&f_)};

        // launch the kernel
        if(cudaError_t error = cudaLaunchKernel(ptr_to_kernel, grid_dim_, block_dim_, ptr_to_arg, shared_memory_size_, stream_))
        {
          detail::throw_on_error(error, "kernel_launch::start: CUDA error after cudaLaunchKernel");
        }
#  else
        // copy the parameter
        void* ptr_to_arg = cudaGetParameterBuffer(std::alignment_of<Function>::value, sizeof(Function));
        std::memcpy(ptr_to_arg, &f_, sizeof(Function));

        // launch the kernel
        if(cudaError_t error = cudaLaunchDevice(ptr_to_kernel, ptr_to_arg, grid_dim_, block_dim_, shared_memory_size_, stream_))
        {
          detail::throw_on_error(error, "kernel_launch::start: CUDA error after cudaLaunchDevice");
        }
#  endif
      }
#else
      detail::throw_on_error(cudaErrorNotSupported, "kernel_launch::start");
#endif

      // switch back to the original device
      set_device(old_device);
    }

    CUDEX_ANNOTATION
    cudaStream_t stream() const
    {
      return stream_;
    }

  private:
    template<class Arg>
    CUDEX_ANNOTATION
    static void workaround_unused_variable_warning(Arg&&) noexcept {}

    CUDEX_ANNOTATION
    static int set_device(int device)
    {
      int result = -1;

#if CUDEX_HAS_CUDART
      if(cudaError_t error = cudaGetDevice(&result))
      {
        detail::throw_on_error(error, "kernel_launch::set_device: CUDA error after cudaGetDevice");
      }

#ifndef __CUDA_ARCH__
      if(cudaError_t error = cudaSetDevice(device))
      {
        detail::throw_on_error(error, "kernel_launch::set_device: CUDA error after cudaSetDevice");
      }
#else
      if(device != result)
      {
        detail::throw_on_error(cudaErrorNotSupported, "kernel_launch::set_device: the requested device must be the same as the current device in __device__ code");
      }
#endif

#else
      detail::throw_on_error(cudaErrorNotSupported, "kernel_launch::set_device");
#endif

      return result;
    }

    Function f_;
    dim3 grid_dim_;
    dim3 block_dim_;
    std::size_t shared_memory_size_;
    cudaStream_t stream_;
    int device_;
};


template<class Function>
CUDEX_ANNOTATION
kernel_launch<Function> make_kernel_launch(Function f, dim3 grid_dim, dim3 block_dim, std::size_t shared_memory_size, cudaStream_t stream, int device) noexcept
{
  return {f, grid_dim, block_dim, shared_memory_size, stream, device};
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

