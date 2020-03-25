#include <cassert>
#include <cudex/detail/kernel_sender.hpp>

#ifdef __CUDACC__


__managed__ int result;

__host__ __device__
void test()
{
  result = -1;
  int expected = 13;

  auto f = [=] __device__ ()
  {
    result = expected;
  };

  cudaEvent_t event{};
  assert(cudaEventCreateWithFlags(&event, cudaEventDefault) == cudaSuccess);

  cudex::detail::make_kernel_sender(f, dim3(1), dim3(1), 0, 0, 0).connect(event).start();

#ifndef __CUDA_ARCH__
  assert(cudaEventSynchronize(event) == cudaSuccess);
#else
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif

  assert(expected == result);

  assert(cudaEventDestroy(event) == cudaSuccess);
}


__global__ void global_function()
{
  test();
}


#endif // __CUDACC__


void test_kernel_sender()
{
#if __CUDACC__
  // test from host
  test();

  // test from device
  global_function<<<1,1>>>();
#endif
}

