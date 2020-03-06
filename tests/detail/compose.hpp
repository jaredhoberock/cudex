#include <cassert>
#include <cudex/detail/functional/compose.hpp>
#include <utility>

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


struct mutable_invocable1
{
  int addend;

  __host__ __device__
  int operator()(int arg)
  {
    return addend + arg;
  }

  __host__ __device__
  int operator()()
  {
    return addend;
  }
};


struct mutable_invocable2
{
  __host__ __device__
  int operator()(int x, int y)
  {
    return x + y;
  }

  __host__ __device__
  void operator()(int) {}
};


struct const_invocable1
{
  int addend;

  __host__ __device__
  int operator()(int arg) const
  {
    return addend + arg;
  }

  __host__ __device__
  int operator()() const
  {
    return addend;
  }
};


struct const_invocable2
{
  __host__ __device__
  int operator()(int x, int y) const
  {
    return x + y;
  }

  __host__ __device__
  void operator()(int) const {}
};


struct rvalue_invocable1
{
  int addend;

  __host__ __device__
  int operator()(int arg) &&
  {
    return addend + arg;
  }

  __host__ __device__
  int operator()() &&
  {
    return addend;
  }
};


struct rvalue_invocable2
{
  __host__ __device__
  int operator()(int x, int y) &&
  {
    return x + y;
  }

  __host__ __device__
  void operator()(int) && {}
};


struct copyable_invocable1
{
  int addend;

  __host__ __device__
  int operator()(int arg)
  {
    return addend + arg;
  }
};


struct copyable_invocable2
{
  __host__ __device__
  int operator()(int x, int y)
  {
    return x + y;
  }
};


struct move_only_invocable1
{
  move_only_invocable1(move_only_invocable1&&) = default;

  int addend;

  __host__ __device__
  int operator()(int arg)
  {
    return addend + arg;
  }
};


struct move_only_invocable2
{
  move_only_invocable2() = default;

  move_only_invocable2(move_only_invocable2&&) = default;

  __host__ __device__
  int operator()(int x, int y)
  {
    return x + y;
  }
};


__host__ __device__
void test()
{
  using namespace cudex::detail;

  {
    int expected = 13 + 7 + 42;

    mutable_invocable1 f{13};
    mutable_invocable2 g;

    // test g returning non-void case
    auto f_of_g = compose(f,g);
    int result = f_of_g(7,42);

    assert(expected == result);

    // test g returning void case
    expected = 13;
    result = f_of_g(7);
    assert(expected == result);
  }

  {
    int expected = 13 + 7 + 42;

    const_invocable1 f{13};
    const_invocable2 g;

    // test g returning non-void case
    auto f_of_g = compose(f,g);
    int result = cref(f_of_g)(7,42);

    assert(expected == result);

    // test g returning void case
    expected = 13;
    result = f_of_g(7);
    assert(expected == result);
  }

  {
    int expected = 13 + 7 + 42;

    rvalue_invocable1 f{13};
    rvalue_invocable2 g;

    // test g returning non-void case
    auto f_of_g = compose(f,g);
    int result = std::move(f_of_g)(7,42);

    assert(expected == result);

    // test g returning void case
    expected = 13;
    result = std::move(f_of_g)(7);
    assert(expected == result);
  }

  {
    // test move-only invocables
    int expected = 13 + 7 + 42;

    move_only_invocable1 f{13};
    move_only_invocable2 g;

    auto f_of_g = compose(std::move(f), std::move(g));

    // test move ctor
    auto f_of_g_moved = std::move(f_of_g);
    int result = f_of_g_moved(7,42);

    assert(expected == result);
  }

  {
    // test copyable invocables
    int expected = 13 + 7 + 42;

    copyable_invocable1 f{13};
    copyable_invocable2 g;

    auto f_of_g = compose(f, g);

    // test copy ctor
    auto f_of_g_copy = f_of_g;
    int result = f_of_g_copy(7,42);

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


void test_compose()
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

