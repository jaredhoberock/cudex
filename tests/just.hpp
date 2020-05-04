#include <cassert>
#include <cstring>
#include <cudex/executor/inline_executor.hpp>
#include <cudex/just.hpp>
#include <exception>
#include <utility>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif


__managed__ int result;


struct move_only
{
  int value;

  move_only(move_only&&) = default;
};


struct my_receiver
{
  __host__ __device__
  void set_value(int value)
  {
    result = value;
  }

  __host__ __device__
  void set_value(move_only&& value)
  {
    result = value.value;
  }

  void set_error(std::exception_ptr) {}

  __host__ __device__
  void set_done() noexcept {}
};


__host__ __device__
void test()
{
  using namespace cudex;

  {
    // test copyable type
    
    result = 0;
    int expected = 13;

    my_receiver r;

    just(expected).connect(std::move(r)).start();

    assert(expected == result);
  }

  {
    // test move-only type

    result = 0;
    int expected = 13;

    my_receiver r;

    just(move_only{expected}).connect(std::move(r)).start();

    assert(expected == result);
  }
}


template<class F>
__global__ void device_invoke_kernel(F f)
{
  f();
}


template<class F>
void device_invoke(F f)
{
#if defined(__CUDACC__)
  device_invoke_kernel<<<1,1>>>(f);
  assert(cudaDeviceSynchronize() == cudaSuccess);
#else
  // device invocations are not supported
  assert(0);
#endif
}


void test_just()
{
  test();

#ifdef __CUDACC__
  device_invoke([] __device__ ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

