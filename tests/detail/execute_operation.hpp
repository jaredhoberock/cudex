#include <cassert>
#include <cudex/detail/execute_operation.hpp>
#include <cudex/inline_executor.hpp>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


template<class T>
__host__ __device__
const T& cref(const T& ref)
{
  return ref;
}


struct mutable_invocable
{
  int& result;
  int value;

  __host__ __device__
  void operator()()
  {
    result = value;
  }
};


struct const_invocable
{
  int& result;
  int value;

  __host__ __device__
  void operator()() const
  {
    result = value;
  }
};


struct move_invocable
{
  int& result;
  int value;

  __host__ __device__
  void operator()() &&
  {
    result = value;
  }
};


__host__ __device__
void test()
{
  using namespace cudex;
  using namespace cudex::detail;

  inline_executor ex;

  {
    int result = 0;
    int expected = 13;

    mutable_invocable f{result, expected};
    auto op = make_execute_operation(ex, f);

    op.start();

    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    const_invocable f{result, expected};
    auto op = make_execute_operation(ex, f);

    cref(op).start();

    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    const_invocable f{result, expected};
    auto op = make_execute_operation(ex, f);

    std::move(op).start();

    assert(expected == result);
  }
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_execute_operation()
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

