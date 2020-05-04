#include <cassert>
#include <cudex/executor/execute.hpp>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


struct has_execute_member
{
  template<class F>
  __host__ __device__
  auto execute(F&& f) const
    -> decltype(f())
  {
    return f();
  }
};


struct has_execute_free_function {};

template<class F>
__host__ __device__
auto execute(const has_execute_free_function&, F&& f)
  -> decltype(f())
{
  return f();
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


__host__ __device__
void test()
{
  {
    has_execute_member e;

    bool invoked = false;
    cudex::execute(e, [&]{ invoked = true; });
    assert(invoked);
  }

  {
    has_execute_free_function e;

    bool invoked = false;
    cudex::execute(e, [&]{ invoked = true; });
    assert(invoked);
  }
}


void test_execute()
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

