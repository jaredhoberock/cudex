#include <cassert>
#include <cudex/detail/grid.hpp>

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

  cudex::detail::make_grid(f, dim3(1), dim3(1), 0, 0, 0).connect(event).start();

#ifndef __CUDA_ARCH__
  assert(cudaEventSynchronize(event) == cudaSuccess);
#else
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif

  assert(expected == result);

  assert(cudaEventDestroy(event) == cudaSuccess);
}


__global__ void test_kernel()
{
  test();
}


#endif // __CUDACC__


void test_grid()
{
#if __CUDACC__
  // test from host
  test();

  // test from device
  test_kernel<<<1,1>>>();
#endif
}

