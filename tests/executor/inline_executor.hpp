#include <cassert>
#include <cudex/executor/inline_executor.hpp>
#include <cudex/property/blocking.hpp>


namespace ns = cudex;


#ifndef __CUDACC__
#define __host__
#define __device__
#endif


__host__ __device__
void test()
{
  ns::inline_executor ex1;

  int result = 0;
  int expected = 13;

  ex1.execute([&]
  {
    result = expected;
  });

  assert(expected == result);

  ns::inline_executor ex2;

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));

  static_assert(ns::blocking.always == ns::inline_executor::query(ns::blocking), "Error");
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

void test_inline_executor()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

