#include <cassert>
#include <cudex/executor/kernel_executor.hpp>
#include <cudex/property/stream.hpp>


namespace ns = cudex;


struct my_executor_with_stream_member
{
  cudaStream_t stream_;

  cudaStream_t stream() const
  {
    return stream_;
  }

  template<class F>
  void execute(F&&) const noexcept;

  bool operator==(const my_executor_with_stream_member&) const;
  bool operator!=(const my_executor_with_stream_member&) const;
};


template<class Executor>
void test()
{
  cudaStream_t expected{};
  assert(cudaSuccess == cudaStreamCreate(&expected));

  Executor ex{expected};
  assert(expected == query(ex, ns::stream));

  assert(cudaSuccess == cudaStreamDestroy(expected));
}


void test_stream()
{
  test<ns::kernel_executor>();
  test<my_executor_with_stream_member>();
}

