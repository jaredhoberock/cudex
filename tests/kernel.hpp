#include <cassert>
#include <cudex/detail/kernel.hpp>

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

  auto kernel = cudex::detail::make_kernel(f, dim3(1), dim3(1), 0, 0, 0);

  kernel.start();

  cudaDeviceSynchronize();

  assert(result == expected);
}


__global__ void kernel()
{
  test();
}


#endif // __CUDACC__


void test_kernel()
{
#if __CUDACC__
  test();

  kernel<<<1,1>>>();
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

