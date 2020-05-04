#include <cassert>
#include <cudex/executor/stream_executor.hpp>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __managed__
#define __managed__
#endif

#ifndef __global__
#define __global__
#endif


__managed__ int result;


__host__ __device__
void test(cudaStream_t s, int d)
{
  using namespace cudex;

  stream_executor ex1{s, d};

  assert(ex1.stream() == s);
  assert(ex1.device() == d);

  result = 0;
  int expected = 13;

  ex1.execute([=] __device__
  {
    result = expected;
  });

#ifndef __CUDA_ARCH__
  assert(cudaStreamSynchronize(s) == cudaSuccess);
#else
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif

  assert(expected == result);

  stream_executor ex2{s};

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


__host__ __device__
void test_on_default_stream()
{
  test(cudaStream_t{}, 0);
}


__host__ __device__
void test_on_new_stream()
{
  cudaStream_t s{};
  assert(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

  test(s, 0);

  assert(cudaStreamDestroy(s) == cudaSuccess);
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_stream_executor()
{
#ifdef __CUDACC__
  test_on_default_stream();
  test_on_new_stream();

  device_invoke<<<1,1>>>([] __device__ ()
  {
    test_on_default_stream();
    test_on_new_stream();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

