#include <cassert>
#include <utility>
#include <cudex/executor/executor_shape.hpp>
#include <cudex/executor/inline_executor.hpp>
#include <cudex/executor/kernel_executor.hpp>


namespace ns = cudex;


#ifndef __host__
#define __host__
#define __device__
#endif


struct has_nested_coordinate_type : ns::inline_executor
{
  using coordinate_type = std::pair<int,int>;
};


__host__ __device__
void test()
{
  static_assert(std::is_same<std::pair<int,int>, ns::executor_shape_t<has_nested_coordinate_type>>::value, "Error.");
  static_assert(std::is_same<std::size_t, ns::executor_shape_t<ns::inline_executor>>::value, "Error.");
  static_assert(std::is_same<ns::kernel_executor::coordinate_type, ns::executor_shape_t<ns::kernel_executor>>::value, "Error.");
}


void test_executor_shape()
{
  test();
}

