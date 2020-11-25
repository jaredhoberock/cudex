#include <cassert>
#include <cudex/executor/get_executor.hpp>
#include <cudex/executor/inline_executor.hpp>
#include <cudex/stream_pool.hpp>

namespace ns = cudex;


struct has_executor_member_function
{
  ns::inline_executor executor() const
  {
    return {};
  }
};


struct has_get_executor_free_function {};


ns::inline_executor get_executor(has_get_executor_free_function)
{
  return {};
}


template<class Arg, class Executor>
void test(Arg&& arg, Executor expected)
{
  auto ex = ns::get_executor(std::forward<Arg>(arg));
  assert(expected == ex);
}

void test_get_executor()
{
  ns::static_stream_pool pool{0,1};
  test(pool, pool.executor());
  test(has_executor_member_function{}, ns::inline_executor{});
  test(has_get_executor_free_function{}, ns::inline_executor{});
}

