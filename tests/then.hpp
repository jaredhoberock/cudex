#include <cassert>
#include <cstring>
#include <cudex/executor/inline_executor.hpp>
#include <cudex/just.hpp>
#include <cudex/then.hpp>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __managed__
#define __global__
#endif


__managed__ int result;


template<class Function>
class move_only_function
{
  public:
    __host__ __device__
    move_only_function(Function f)
      : f_(f)
    {}

    move_only_function(const move_only_function&) = delete;

    move_only_function(move_only_function&&) = default;

    __host__ __device__
    int operator()(int arg) const
    {
      return f_(arg);
    }

  private:
    Function f_;
};


template<class Function>
__host__ __device__
move_only_function<Function> make_move_only_function(Function f)
{
  return {f};
}


struct my_receiver
{
  __host__ __device__
  void set_value(int value)
  {
    result = value;
  }

  void set_error(std::exception_ptr) {}

  __host__ __device__
  void set_done() noexcept {}
};


__host__ __device__
void test_copyable_continuation()
{
  using namespace cudex;

  result = 0;
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  my_receiver r;

  then(just(arg1), [=] (int arg1) { return arg1 + arg2; }).connect(std::move(r)).start();

  assert(expected == result);
}


__host__ __device__
void test_move_only_continuation()
{
  using namespace cudex;

  result = 0;
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  my_receiver r;

  auto continuation = make_move_only_function([=] (int arg1) { return arg1 + arg2; });

  then(just(arg1), std::move(continuation)).connect(std::move(r)).start();

  assert(expected == result);
}


struct my_sender_with_then_member_function
{
  int arg;

  template<class Function>
  __host__ __device__
  auto then(Function continuation) &&
    -> decltype(cudex::just(arg).then(continuation))
  {
    return cudex::just(arg).then(continuation);
  }
};


__host__ __device__
void test_sender_with_then_member_function()
{
  result = 0;
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  my_receiver r;

  my_sender_with_then_member_function s{arg1};

  cudex::then(std::move(s), [=](int arg1) {return arg1 + arg2;}).connect(std::move(r)).start();

  assert(expected == result);
}


struct my_sender_with_then_free_function
{
  int arg;
};


template<class Function>
__host__ __device__
auto then(my_sender_with_then_free_function&& s, Function continuation)
  -> decltype(cudex::just(s.arg).then(continuation))
{
  return cudex::just(s.arg).then(continuation);
}


__host__ __device__
void test_sender_with_then_free_function()
{
  result = 0;
  int arg1 = 13;
  int arg2 = 7;
  int expected = arg1 + arg2;

  my_receiver r;

  my_sender_with_then_member_function s{arg1};

  cudex::then(std::move(s), [=](int arg1) {return arg1 + arg2;}).connect(std::move(r)).start();

  assert(expected == result);
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


void test_then()
{
  test_copyable_continuation();
  test_move_only_continuation();
  test_sender_with_then_member_function();
  test_sender_with_then_free_function();

#ifdef __CUDACC__
  device_invoke([] __device__ ()
  {
    test_copyable_continuation();
    test_move_only_continuation();
    test_sender_with_then_member_function();
    test_sender_with_then_free_function();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

