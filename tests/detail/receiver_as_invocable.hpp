#include <cassert>
#include <cudex/detail/receiver_as_invocable.hpp>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

struct my_receiver
{
  int& result;

  __host__ __device__
  void set_value(int value) noexcept
  {
    result = value;
  }

  void set_error(std::exception_ptr) {}

  void set_done() noexcept {}
};


__host__ __device__
void test()
{
  using namespace cudex::detail;

  int result = 0;
  int expected = 13;

  my_receiver r{result};

  auto f = as_invocable(std::move(r));

  f(expected);

  assert(expected == result);

  auto g = as_invocable(std::move(r));

  // test copy ctor
  auto copy = g;
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

void test_receiver_as_invocable()
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

