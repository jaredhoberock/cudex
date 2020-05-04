#include <cassert>
#include <cstring>
#include <cudex/executor/inline_executor.hpp>
#include <cudex/just_on.hpp>
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


template<class Executor>
__host__ __device__
void test_copyable(Executor ex)
{
  using namespace cudex;

  result = 0;
  int expected = 13;

  my_receiver r;

  just_on(ex, expected).connect(std::move(r)).start();

  assert(expected == result);
}


template<class Executor>
__host__ __device__
void test_move_only(Executor ex)
{
  using namespace cudex;

  result = 0;
  int expected = 13;

  my_receiver r;

  just_on(ex, move_only{expected}).connect(std::move(r)).start();

  assert(expected == result);
}


struct my_executor_with_just_on_member_function : cudex::inline_executor
{
  template<class T>
  __host__ __device__
  auto just_on(T&& value) const
    -> decltype(cudex::just_on(cudex::inline_executor(), std::forward<T>(value)))
  {
    return cudex::invoke_on(cudex::inline_executor(), std::forward<T>(value));
  }
};


struct my_executor_with_just_on_free_function : cudex::inline_executor {};


template<class T>
__host__ __device__
auto just_on(my_executor_with_just_on_free_function, T&& value)
  -> decltype(cudex::just_on(cudex::inline_executor{}, std::forward<T>(value)))
{
  return cudex::just_on(cudex::inline_executor{}, std::forward<T>(value));
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


void test_just_on()
{
  test_copyable(cudex::inline_executor{});
  test_copyable(my_executor_with_just_on_member_function{});
  test_copyable(my_executor_with_just_on_free_function{});

  test_move_only(cudex::inline_executor{});
  test_move_only(my_executor_with_just_on_member_function{});
  test_move_only(my_executor_with_just_on_free_function{});

#ifdef __CUDACC__
  test_copyable(gpu_executor{});

  device_invoke([] __device__ ()
  {
    test_copyable(cudex::inline_executor{});
    test_copyable(my_executor_with_just_on_member_function{});
    test_copyable(my_executor_with_just_on_free_function{});

    test_move_only(cudex::inline_executor{});
    test_move_only(my_executor_with_just_on_member_function{});
    test_move_only(my_executor_with_just_on_free_function{});

    test_copyable(gpu_executor{});
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

