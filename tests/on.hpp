#include <cassert>
#include <cstring>
#include <cudex/just.hpp>
#include <cudex/on.hpp>
#include <cudex/then.hpp>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif


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


__managed__ int num_calls_to_my_executor_execute = 0;

struct my_executor
{
  __host__ __device__
  bool operator==(const my_executor&) const { return true; }

  __host__ __device__
  bool operator!=(const my_executor&) const { return false; }

  __host__ __device__
  static int num_calls_to_execute()
  {
    return num_calls_to_my_executor_execute;
  }

  __host__ __device__
  static void initialize_num_calls_to_execute()
  {
    num_calls_to_my_executor_execute = 0;
  }

  template<class Function>
  __host__ __device__
  void execute(Function&& f) const noexcept
  {
    std::forward<Function>(f)();
    ++num_calls_to_my_executor_execute;
  }
};


__managed__ int num_calls_to_my_gpu_executor_execute = 0;

struct my_gpu_executor
{
  __host__ __device__
  bool operator==(const my_gpu_executor&) const { return true; }

  __host__ __device__
  bool operator!=(const my_gpu_executor&) const { return false; }

  __host__ __device__
  static int num_calls_to_execute()
  {
    return num_calls_to_my_gpu_executor_execute;
  }

  __host__ __device__
  static void initialize_num_calls_to_execute()
  {
    num_calls_to_my_gpu_executor_execute = 0;
  }

  template<class Function>
  __host__ __device__
  void execute(Function f) const noexcept
  {
    device_invoke(f);
    ++num_calls_to_my_gpu_executor_execute;
  }
};


__managed__ int result;


struct my_receiver
{
  __host__ __device__
  void set_value(int value)
  {
    result = value;
  }

  void set_error(std::exception_ptr) {}

  __host__ __device__
  void set_done() noexcept {}
};


template<class Executor>
__host__ __device__
void test(Executor ex)
{
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  result = 0;
  Executor::initialize_num_calls_to_execute();

  cudex::just(arg1)
    .then([=] __host__ __device__ (int arg1)
     {
       return arg1 + arg2;
     })
    .on(ex)
    .submit(my_receiver{});

  assert(result == expected);
  assert(1 == my_executor::num_calls_to_execute());
}


void test_on()
{
  test(my_executor{});

#if __CUDACC__
  test(my_gpu_executor{});

  device_invoke([] __device__ ()
  {
    test(my_executor{});
    test(my_gpu_executor{});
  });
#endif
}

