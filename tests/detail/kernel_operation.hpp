#include <cassert>
#include <cudex/detail/kernel_operation.hpp>

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

  auto op = cudex::detail::make_kernel_operation(f, dim3(1), dim3(1), 0, 0, 0);

  op.start();

  cudaDeviceSynchronize();

  assert(result == expected);
}


__global__ void global_function()
{
  test();
}


#endif // __CUDACC__


void test_kernel_operation()
{
#if __CUDACC__
  test();

  global_function<<<1,1>>>();
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

