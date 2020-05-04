#include <cassert>
#include <cudex/detail/execute_operation.hpp>
#include <cudex/executor/inline_executor.hpp>

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


struct rvalue_invocable
{
  int& result;
  int value;

  __host__ __device__
  void operator()() &&
  {
    result = value;
  }
};


struct copyable_invocable
{
  int& result;
  int value;

  copyable_invocable(const copyable_invocable&) = default;

  __host__ __device__
  void operator()()
  {
    result = value;
  }
};


struct move_only_invocable
{
  int& result;
  int value;

  move_only_invocable(move_only_invocable&&) = default;

  __host__ __device__
  void operator()()
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

    rvalue_invocable f{result, expected};
    auto op = make_execute_operation(ex, f);

    std::move(op).start();

    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    copyable_invocable f{result, expected};
    auto op = make_execute_operation(ex, f);

    auto op_copy = op;

    op_copy.start();

    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    move_only_invocable f{result, expected};
    auto op = make_execute_operation(ex, std::move(f));

    auto op_moved = std::move(op);

    op_moved.start();

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

