#include <cassert>
#include <cudex/executor/inline_executor.hpp>
#include <cudex/property/blocking.hpp>


#ifndef __host__
#define __host__
#define __device__
#define __global__
#endif


namespace ns = cudex;


struct my_executor_with_static_blocking_query
  : ns::inline_executor
{
  __host__ __device__
  constexpr static ns::blocking_t query(ns::blocking_t)
  {
    return ns::blocking.always;
  }
};


struct my_executor_with_dynamic_blocking_query
  : ns::inline_executor
{
  __host__ __device__
  ns::blocking_t query(ns::blocking_t) const
  {
    return ns::blocking.always;
  }
};


__host__ __device__
void test()
{
  {
    // test static blocking query
    static_assert(ns::blocking.always == my_executor_with_static_blocking_query::query(ns::blocking), "Error");
  }

  {
    // test dynamic blocking query
    my_executor_with_dynamic_blocking_query ex;
    assert(ns::blocking.always == ex.query(ns::blocking));
  }
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


__host__ __device__
void test_blocking()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test();
  });
  assert(cudaSuccess == cudaDeviceSynchronize());
#endif
}

