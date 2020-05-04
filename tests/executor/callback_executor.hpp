#include <cassert>
#include <cudex/executor/callback_executor.hpp>


int result;


void test(cudaStream_t s)
{
  using namespace cudex;

  callback_executor ex1{s};

  assert(ex1.stream() == s);

  result = 0;
  int expected = 13;

  ex1.execute([=]
  {
    result = expected;
  });

  assert(cudaStreamSynchronize(s) == cudaSuccess);

  assert(expected == result);

  callback_executor ex2{s};

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


void test_on_default_stream()
{
  test(cudaStream_t{});
}


void test_on_new_stream()
{
  cudaStream_t s{};
  assert(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

  test(s);

  assert(cudaStreamDestroy(s) == cudaSuccess);
}


void test_callback_executor()
{
  test_on_default_stream();
  test_on_new_stream();
}

