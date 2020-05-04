#include <cassert>
#include <iostream>
#include <cudex/executor/bulk_execute.hpp>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


struct has_bulk_execute_member_function
{
  template<class F>
  __host__ __device__
  void bulk_execute(F&& f, std::size_t n) const
  {
    for(size_t i = 0; i < n; ++i)
    {
      f(i);
    }
  }
};


struct has_bulk_execute_free_function {};

template<class F>
__host__ __device__
void bulk_execute(const has_bulk_execute_free_function&, F&& f, std::size_t n)
{
  return has_bulk_execute_member_function().bulk_execute(std::forward<F>(f), n);
}


struct has_execute_member_function
{
  template<class F>
  __host__ __device__
  void execute(F&& f) const
  {
    f();
  }

  __host__ __device__
  bool operator==(const has_execute_member_function&) const
  {
    return true;
  }
};


struct has_execute_member_function_and_static_unsequenced_guarantee
{
  template<class F>
  __host__ __device__
  void execute(F&& f) const
  {
    f();
  }

  __host__ __device__
  bool operator==(const has_execute_member_function_and_static_unsequenced_guarantee&) const
  {
    return true;
  }

  __host__ __device__
  constexpr static cudex::bulk_guarantee_t::unsequenced_t query(cudex::bulk_guarantee_t)
  {
    return cudex::bulk_guarantee.unsequenced;
  }
};


struct has_execute_free_function
{
  __host__ __device__
  bool operator==(const has_execute_free_function&) const
  {
    return true;
  }
};

template<class F>
__host__ __device__
void execute(const has_execute_free_function&, F&& f)
{
  f();
}


struct has_execute_free_function_and_static_unsequenced_guarantee
{
  __host__ __device__
  bool operator==(const has_execute_free_function_and_static_unsequenced_guarantee&) const
  {
    return true;
  }
};

template<class F>
__host__ __device__
void execute(const has_execute_free_function_and_static_unsequenced_guarantee&, F&& f)
{
  f();
}


__host__ __device__
size_t factorial(size_t n)
{
  size_t result = 1;
  for(size_t i = 1; i <= n; ++i)
  {
    result *= i;
  }

  return result;
}


template<class Function>
struct noncopyable_invocable
{
  mutable Function f;

  __host__ __device__
  noncopyable_invocable(const noncopyable_invocable&) = delete;

  __host__ __device__
  void operator()(size_t i) const
  {
    f(i);
  }
};


template<class F>
__host__ __device__
noncopyable_invocable<typename std::decay<F>::type> make_noncopyable(F&& f)
{
  return {std::forward<F>(f)};
}


template<class Function>
struct nonmovable_invocable
{
  mutable Function f;

  nonmovable_invocable(nonmovable_invocable&&) = delete;

  __host__ __device__
  void operator()(size_t i) const
  {
    f(i);
  }
};


template<class F>
__host__ __device__
nonmovable_invocable<typename std::decay<F>::type> make_nonmovable(F&& f)
{
  return {std::forward<F>(f)};
}


template<class Executor>
__host__ __device__
void test(const Executor& e)
{
  size_t n = 7;
  
  // test copyable invocable
  size_t result = 1;
  cudex::bulk_execute(e, [&](size_t i)
  {
    result *= (i + 1);
  }, n);
  
  assert(result == factorial(n));
  
  // test non-copyable invocable
  result = 1;
  cudex::bulk_execute(e, make_noncopyable([&](size_t i)
  {
    result *= (i + 1);
  }), n);
  
  assert(result == factorial(n));

  // test non-movable invocable
  result = 1;
  cudex::bulk_execute(e, make_nonmovable([&](size_t i)
  {
    result *= (i + 1);
  }), n);
  
  assert(result == factorial(n));
}


__host__ __device__
void test()
{
  test(has_bulk_execute_member_function{});

  test(has_bulk_execute_free_function{});

  test(has_execute_member_function{});

  test(has_execute_free_function{});

  test(has_execute_member_function_and_static_unsequenced_guarantee{});
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_bulk_execute()
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

