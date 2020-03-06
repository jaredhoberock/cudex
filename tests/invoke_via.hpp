#include <cassert>
#include <cstring>
#include <cudex/inline_executor.hpp>
#include <cudex/invoke_via.hpp>
#include <exception>
#include <utility>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif


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
  using namespace cudex;

  {
    // test with 0 args

    result = 0;
    int expected = 13;

    my_receiver r;

    auto return_expected = [=] __host__ __device__ { return expected; };
    invoke_via(ex, return_expected).connect(std::move(r)).start();

    assert(expected == result);
  }

  {
    // test with 1 arg

    result = 0;
    int arg = 13;
    int expected = arg;

    my_receiver r;

    auto identity = [] __host__ __device__ (int x){ return x; };
    invoke_via(ex, identity, arg).connect(std::move(r)).start();

    assert(expected == result);
  }

  {
    // test with 2 args

    result = 0;
    int arg1 = 13;
    int arg2 = 7;
    int expected = arg1 + arg2;

    my_receiver r;

    auto plus = [] __host__ __device__ (int x, int y){ return x + y; };
    invoke_via(ex, plus, arg1, arg2).connect(std::move(r)).start();

    assert(expected == result);
  }
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


struct gpu_executor
{
  __host__ __device__
  bool operator==(const gpu_executor&) const { return true; }

  __host__ __device__
  bool operator!=(const gpu_executor&) const { return false; }

  template<class Function>
  __host__ __device__
  void execute(Function f) const noexcept
  {
    device_invoke(f);
  }
};


void test_invoke_via()
{
  test(cudex::inline_executor{});

#ifdef __CUDACC__
  test(gpu_executor{});

  device_invoke([] __device__ ()
  {
    test(cudex::inline_executor{});
    test(gpu_executor{});
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

