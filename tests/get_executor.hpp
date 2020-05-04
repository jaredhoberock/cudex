#include <cassert>
#include <cstring>
#include <cudex/executor/inline_executor.hpp>
#include <cudex/get_executor.hpp>
#include <exception>
#include <utility>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif


struct has_get_executor_member_function
{
  __host__ __device__
  cudex::inline_executor get_executor() const
  {
    return {};
  }
};


struct has_get_executor_free_function {};

__host__ __device__
cudex::inline_executor get_executor(has_get_executor_free_function)
{
  return {};
}


template<class F>
__global__ void device_invoke_kernel(F f)
{
  f();
}


template<class F>
__host__ __device__
void device_invoke(F f)
{
#if defined(__CUDACC__)

#if !defined(__CUDA_ARCH__)
  // __host__ path

  device_invoke_kernel<<<1,1>>>(f);

#else
  // __device__ path

  // workaround restriction on parameters with copy ctors passed to triple chevrons
  void* ptr_to_arg = cudaGetParameterBuffer(std::alignment_of<F>::value, sizeof(F));
  std::memcpy(ptr_to_arg, &f, sizeof(F));

  // launch the kernel
  if(cudaLaunchDevice(&device_invoke_kernel<F>, ptr_to_arg, dim3(1), dim3(1), 0, 0) != cudaSuccess)
  {
    assert(0);
  }
#endif

  assert(cudaDeviceSynchronize() == cudaSuccess);
#else
  // device invocations are not supported
  assert(0);
#endif
}


template<class T>
__host__ __device__
void test(T&& arg)
{
  auto ex = cudex::get_executor(std::forward<T>(arg));

  // assert that the thing we got acts like an executor
  int expected = 13;
  int result = 0;

  ex.execute([&]
  {
    result = expected;
  });

  assert(expected == result);
}


void test_get_executor()
{
  test(has_get_executor_free_function{});
  test(has_get_executor_member_function{});

#ifdef __CUDACC__
  device_invoke([] __device__ ()
  {
    test(has_get_executor_free_function{});
    test(has_get_executor_member_function{});
  });
#endif
}

